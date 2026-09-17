"""
Step 12 清理管理命令单元测试。

范围
----
- CleanupVLMStatesTest: 默认 30 天 / --days 60 / --keep-hits / --dry-run
- CleanupFramesTest: 默认 7 天 / 非日期目录 / --dry-run / MEDIA_ROOT/frames 不存在

技术细节
--------
- 碰 DB 的用例一律 `django.test.TestCase`（独立 test 库 + 自动回滚），并且额外
  用 `_purge()` 按 `t_<uuid8>_` 前缀兜底回收。原因见 config_panel/tests.py 顶部：
  整批测试里若没有 TestCase，Django 会跳过 test 库创建，测试会直接污染真实库；
  而 `cleanup_vlm_states` 是**全局按时间删除**的，落在真实库上就会删用户历史。
- 文件类用例（CleanupFramesTest）用临时 MEDIA_ROOT，不碰 DB。
- cleanup_vlm_states 直接 mock `django.utils.timezone.now` 不行（命令里
  timezone.now() 是 bound name），改用 "手动 -N 秒" 创建老条目（用 .update 绕过 auto_now_add）。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.conf import settings  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import TestCase, override_settings  # noqa: E402

from apps.streaming.models import Camera  # noqa: E402
from apps.vlm.models import NotifyTarget, VLMPromptConfig, VLMCheckState  # noqa: E402

def _test_name(base: str) -> str:
    """测试 cam/prompt name 加 uuid 后缀，避免撞用户 DB unique 约束。"""
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_state(
    *, camera: Camera, prompt: VLMPromptConfig, created_at,
    hit: bool = False,
) -> VLMCheckState:
    """建一条 VLMCheckState 并显式设 created_at（绕开 auto_now_add）。"""
    s = VLMCheckState.objects.create(
        camera=camera,
        prompt_config=prompt,
        ts_list=[],
        window_sec=10,
        hit=hit,
        raw_response="",
        status="",
        img1="", img2="", img3="",
        img1_ts=None, img2_ts=None, img3_ts=None,
        has_target_in_window=False,
        failure_reason="",
        retry_count=0,
    )
    # 直接 UPDATE 写 created_at
    VLMCheckState.objects.filter(pk=s.pk).update(created_at=created_at)
    s.refresh_from_db()
    return s


# ---------------------------------------------------------------------------
# cleanup_vlm_states
# ---------------------------------------------------------------------------
class CleanupVLMStatesTest(TestCase):
    """cleanup_vlm_states 管理命令测试。

    **必须用 `django.test.TestCase`**：这条命令是按时间做**全局删除**的
    （`VLMCheckState.objects.filter(created_at__lt=cutoff).delete()`）。
    原来用 plain unittest 时本类会落到真实库上——等于每次跑测试都可能把用户
    30 天前的真实历史删掉。TestCase 在独立 test 库里跑并回滚，命令只能碰到
    本用例自己造的数据。

    `_purge()` 是二次保险（万一有人用 `python -m unittest` 直接跑）。
    """

    def setUp(self):
        self._tok = f"t_{uuid.uuid4().hex[:8]}_"
        self.addCleanup(self._purge)
        self.cam = Camera.objects.create(
            name=self._tok + "cam_clean_test",
            source_type=Camera.SOURCE_FILE, is_active=False,
        )
        self.prompt = VLMPromptConfig.objects.create(
            name=self._tok + "cleanup_prompt",
            prompt="test", positive_keyword="是",
        )

    def _purge(self):
        """删本用例建的行：prompt（CASCADE 清 VLMCheckState）+ 按 cam 兜底 + cam。"""
        VLMPromptConfig.objects.filter(name__startswith=self._tok).delete()
        VLMCheckState.objects.filter(camera_id=self.cam.pk).delete()
        Camera.objects.filter(pk=self.cam.pk).delete()

    def test_default_days_deletes_31_days_old(self):
        from datetime import timedelta
        from io import StringIO
        from django.utils import timezone

        now = timezone.now()
        old = _make_state(camera=self.cam, prompt=self.prompt,
                          created_at=now - timedelta(days=31))
        recent = _make_state(camera=self.cam, prompt=self.prompt,
                             created_at=now - timedelta(days=29))
        out = StringIO()
        call_command("cleanup_vlm_states", stdout=out)

        # old 被删；recent 保留
        self.assertFalse(VLMCheckState.objects.filter(pk=old.pk).exists())
        self.assertTrue(VLMCheckState.objects.filter(pk=recent.pk).exists())
        # 输出含 deleted
        self.assertIn("deleted", out.getvalue())

    def test_days_60_keeps_everything(self):
        from datetime import timedelta
        from django.utils import timezone

        now = timezone.now()
        old = _make_state(camera=self.cam, prompt=self.prompt,
                          created_at=now - timedelta(days=31))
        try:
            call_command("cleanup_vlm_states", "--days", "60")
        finally:
            pass
        self.assertTrue(VLMCheckState.objects.filter(pk=old.pk).exists())

    def test_keep_hits_preserves_hit_true(self):
        from datetime import timedelta
        from django.utils import timezone

        now = timezone.now()
        hit_old = _make_state(camera=self.cam, prompt=self.prompt,
                              created_at=now - timedelta(days=31), hit=True)
        miss_old = _make_state(camera=self.cam, prompt=self.prompt,
                               created_at=now - timedelta(days=31), hit=False)
        try:
            call_command("cleanup_vlm_states", "--keep-hits")
        finally:
            pass
        # hit_old 保留；miss_old 删
        self.assertTrue(VLMCheckState.objects.filter(pk=hit_old.pk).exists())
        self.assertFalse(VLMCheckState.objects.filter(pk=miss_old.pk).exists())

    def test_dry_run_does_not_delete(self):
        from datetime import timedelta
        from django.utils import timezone
        from io import StringIO

        now = timezone.now()
        old = _make_state(camera=self.cam, prompt=self.prompt,
                          created_at=now - timedelta(days=31))
        out = StringIO()
        try:
            call_command("cleanup_vlm_states", "--dry-run", stdout=out)
        finally:
            pass
        # 仍然存在
        self.assertTrue(VLMCheckState.objects.filter(pk=old.pk).exists())
        # 输出含 "would delete"
        self.assertIn("would delete", out.getvalue())


# ---------------------------------------------------------------------------
# cleanup_frames
# ---------------------------------------------------------------------------
class CleanupFramesTest(unittest.TestCase):

    def setUp(self):
        # 临时 MEDIA_ROOT（每个 case 一个独立目录）
        self.tmpdir = tempfile.mkdtemp(prefix="cleanup_frames_test_")
        self.media_root = Path(self.tmpdir)
        self._override = override_settings(MEDIA_ROOT=str(self.media_root))
        self._override.enable()

    def tearDown(self):
        self._override.disable()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_7_days_deletes_old_dir(self):
        # 旧：远早于 7 天；新：今天（一定落在保留窗口内）。
        # 注意：不要写死"新"日期——写死的日期迟早会变成"旧"。
        # （原用例写死 2026-08-30，到 2026-09-12 已落到 7 天窗口之外 → 断言失败）
        # USE_TZ=False，timezone.now() 是 naive 本地时间，不能用 localdate()。
        from django.utils import timezone
        today = timezone.now().date().isoformat()
        old_dir = self.media_root / "frames" / "2020-01-01"
        new_dir = self.media_root / "frames" / today
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        (old_dir / "cam1_1577836800.jpg").write_bytes(b"x")
        (new_dir / "cam1_1693350000.jpg").write_bytes(b"y")

        from io import StringIO
        out = StringIO()
        call_command("cleanup_frames", stdout=out)

        self.assertFalse(old_dir.exists())
        self.assertTrue(new_dir.exists())

    def test_non_date_dir_is_skipped(self):
        from io import StringIO

        frames_dir = self.media_root / "frames"
        frames_dir.mkdir(parents=True)
        # 非日期目录 + 非目录文件
        (frames_dir / "random.txt").write_text("hi")
        (frames_dir / "notadate").mkdir()
        # 一个真日期目录（远期）
        old_dir = frames_dir / "2020-01-01"
        old_dir.mkdir()
        (old_dir / "cam1_1.jpg").write_bytes(b"x")

        out = StringIO()
        call_command("cleanup_frames", stdout=out)
        # random.txt 还在
        self.assertTrue((frames_dir / "random.txt").exists())
        self.assertTrue((frames_dir / "notadate").exists())
        # 远期日期目录被删
        self.assertFalse(old_dir.exists())

    def test_dry_run_does_not_delete(self):
        from io import StringIO

        old_dir = self.media_root / "frames" / "2020-01-01"
        old_dir.mkdir(parents=True)
        (old_dir / "cam1_1.jpg").write_bytes(b"x")

        out = StringIO()
        call_command("cleanup_frames", "--dry-run", stdout=out)
        # 仍然存在
        self.assertTrue(old_dir.exists())
        self.assertIn("would delete", out.getvalue())

    def test_missing_frames_dir_is_noop(self):
        # tmp dir 里完全不建 frames 目录
        from io import StringIO

        out = StringIO()
        call_command("cleanup_frames", stdout=out)
        self.assertIn("no frames dir", out.getvalue())


# ---------------------------------------------------------------------------
# cleanup_test_residue
# ---------------------------------------------------------------------------
class CleanupTestResidueTest(TestCase):
    """清理回归测试残留：只删 `t_<8hex>_` 前缀，绝不碰用户数据。"""

    USER_CAM_NAME = "用户摄像头_不该被删"

    def setUp(self):
        self._tok = f"t_{uuid.uuid4().hex[:8]}_"
        self.addCleanup(self._purge)
        self.cam = Camera.objects.create(
            name=self._tok + "卧房", source_type=Camera.SOURCE_FILE, is_active=False,
        )
        self.prompt = VLMPromptConfig.objects.create(
            name=self._tok + "p", prompt="x", positive_keyword="是",
        )
        self.target = NotifyTarget.objects.create(
            name=self._tok + "dad", kind="mobile_app", target_id="mobile_app_dad",
        )
        # 对照组：名字不像测试残留（uuid 前缀），不能被删
        self.user_cam = Camera.objects.create(
            name=self.USER_CAM_NAME, source_type=Camera.SOURCE_FILE, is_active=False,
        )

    def _purge(self):
        VLMPromptConfig.objects.filter(name__startswith=self._tok).delete()
        NotifyTarget.objects.filter(name__startswith=self._tok).delete()
        Camera.objects.filter(name__startswith=self._tok).delete()
        Camera.objects.filter(name=self.USER_CAM_NAME).delete()

    def test_dry_run_reports_but_keeps_rows(self):
        from io import StringIO

        out = StringIO()
        call_command("cleanup_test_residue", stdout=out)
        self.assertIn("DRY-RUN", out.getvalue())
        self.assertTrue(Camera.objects.filter(pk=self.cam.pk).exists())
        self.assertTrue(VLMPromptConfig.objects.filter(pk=self.prompt.pk).exists())
        self.assertTrue(NotifyTarget.objects.filter(pk=self.target.pk).exists())

    def test_yes_deletes_residue_only(self):
        from io import StringIO

        out = StringIO()
        call_command("cleanup_test_residue", "--yes", stdout=out)
        # 残留被清
        self.assertFalse(Camera.objects.filter(pk=self.cam.pk).exists())
        self.assertFalse(VLMPromptConfig.objects.filter(pk=self.prompt.pk).exists())
        self.assertFalse(NotifyTarget.objects.filter(pk=self.target.pk).exists())
        # 用户数据不动
        self.assertTrue(Camera.objects.filter(pk=self.user_cam.pk).exists())

    def test_nothing_to_clean(self):
        from io import StringIO

        # 先清掉本用例的残留，再跑一次 → 应报"没有残留"
        VLMPromptConfig.objects.filter(name__startswith=self._tok).delete()
        NotifyTarget.objects.filter(name__startswith=self._tok).delete()
        Camera.objects.filter(name__startswith=self._tok).delete()

        out = StringIO()
        call_command("cleanup_test_residue", "--yes", stdout=out)
        self.assertIn("没有残留", out.getvalue())


if __name__ == "__main__":
    unittest.main()
