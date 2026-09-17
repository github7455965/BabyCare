"""Dashboard 视图测试。

范围
----
- HelperTest：_media_url_for 纯函数
- EventsListViewTest：空表 / 1 条 / 过滤器（cam_id / prompt_id / hit=1|0 / kind（B11）
  / start_dt / end_dt（B9））/ 分页 / 非法值忽略
- EventDetailViewTest：存在 / 不存在 404 / 中文同义词（B7）/ ts_list 中文格式（B10）
- LogsViewTest：200 + "暂未实装"
- ImageUrlAttachmentTest：MEDIA_ROOT 内/外路径

约定
----
- VLMCheckState.img1/2/3 存绝对路径（Step 12 决策）；
  这里直接用 Path(MEDIA_ROOT)/frames/<date>/<cam>_<ts>.jpg 写入
- B9 时间过滤：用 VLMCheckState.objects.filter(id__in=...).update(detected_at=...)
  绕过 auto_now_add；test 中显式 update 三条为 today / yesterday / 2 days ago
"""
import os
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.conf import settings  # noqa: E402
from django.test import Client, TestCase  # noqa: E402
from django.utils import timezone  # noqa: E402

from apps.dashboard.views import _media_url_for  # noqa: E402
from apps.streaming.models import Camera  # noqa: E402
from apps.vlm.models import VLMCheckState, VLMPromptConfig  # noqa: E402

def _test_name(base: str) -> str:
    """测试 cam/prompt name 加 uuid 后缀，避免撞用户 DB unique 约束。"""
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


class _Base(TestCase):
    """用 Django TestCase（自动事务回滚）而非 unittest.TestCase，避免污染真实 DB。

    每个测试 setUp 后启动 savepoint；tearDown 自动 rollback，fixture 写的 state
    不会泄漏到真实 DB；之前用 unittest.TestCase 导致 8553+ state 累积让
    filter 测试拿到全表而非 fixture。
    """

    def setUp(self):
        self._orig_allowed = settings.ALLOWED_HOSTS
        settings.ALLOWED_HOSTS = list(self._orig_allowed) + ["testserver"]
        self.addCleanup(setattr, settings, "ALLOWED_HOSTS", self._orig_allowed)

        self.client = Client()
        patcher = patch("apps.vlm.signals._get_manager", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)


def _make_state(cam, prompt, hit=False, **overrides):
    defaults = {
        "camera": cam,
        "prompt_config": prompt,
        "ts_list": [1, 2, 3],
        "window_sec": 3,
        "hit": hit,
    }
    defaults.update(overrides)
    return VLMCheckState.objects.create(**defaults)


class HelperTest(unittest.TestCase):
    """_media_url_for 纯函数测试（不走 DB）。"""

    def test_empty(self):
        self.assertEqual(_media_url_for(""), "")

    def test_inside_media_root(self):
        media_root = settings.MEDIA_ROOT
        abs_path = str(Path(media_root) / "frames" / "2026-01-01" / "1_1700000000.jpg")
        expected = settings.MEDIA_URL + "frames/2026-01-01/1_1700000000.jpg"
        self.assertEqual(_media_url_for(abs_path), expected)

    def test_outside_media_root(self):
        self.assertEqual(_media_url_for("D:\\elsewhere\\x.jpg"), "")

    def test_windows_backslash_replaced(self):
        media_root = settings.MEDIA_ROOT
        abs_path = str(Path(media_root) / "frames" / "2026-01-01" / "2_1700000000.jpg")
        result = _media_url_for(abs_path)
        self.assertNotIn("\\", result.replace(settings.MEDIA_URL, "", 1))


class EventsListViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p1 = VLMPromptConfig.objects.create(
            name=_test_name("prompt1"), prompt="p", positive_keyword="是",
        )
        self.p2 = VLMPromptConfig.objects.create(
            name=_test_name("prompt2"), prompt="p2", positive_keyword="是",
        )

    def test_empty(self):
        resp = self.client.get("/events/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("暂无".encode(), resp.content)

    def test_with_one(self):
        _make_state(self.cam, self.p1, hit=True)
        resp = self.client.get("/events/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("prompt1".encode(), resp.content)
        self.assertIn("cam1".encode(), resp.content)

    def test_filter_cam_id(self):
        cam2 = Camera.objects.create(name=_test_name("cam2"), source_type=Camera.SOURCE_FILE, is_active=False)
        _make_state(self.cam, self.p1)
        s2 = _make_state(cam2, self.p1)
        resp = self.client.get(f"/events/?cam_id={cam2.id}")
        self.assertEqual(resp.status_code, 200)
        # 用 page.object_list 校验
        page = resp.context_data["page"]
        ids = [s.camera_id for s in page.object_list]
        self.assertEqual(ids, [cam2.id])

    def test_filter_prompt_id(self):
        _make_state(self.cam, self.p1)
        _make_state(self.cam, self.p2)
        resp = self.client.get(f"/events/?prompt_id={self.p2.id}")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        self.assertEqual(len(page.object_list), 1)
        self.assertEqual(page.object_list[0].prompt_config_id, self.p2.id)

    def test_filter_hit_1(self):
        _make_state(self.cam, self.p1, hit=True)
        _make_state(self.cam, self.p1, hit=False)
        resp = self.client.get("/events/?hit=1")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        self.assertEqual(page.object_list.count(), 1)
        self.assertTrue(page.object_list[0].hit)

    def test_filter_hit_0(self):
        _make_state(self.cam, self.p1, hit=True)
        _make_state(self.cam, self.p1, hit=False)
        resp = self.client.get("/events/?hit=0")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        self.assertEqual(page.object_list.count(), 1)
        self.assertFalse(page.object_list[0].hit)

    def test_filter_hit_invalid_ignored(self):
        _make_state(self.cam, self.p1)
        resp = self.client.get("/events/?hit=garbage")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context_data["page"].object_list.count(), 1)

    def test_filter_cam_id_non_digit_ignored(self):
        _make_state(self.cam, self.p1)
        resp = self.client.get("/events/?cam_id=abc")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context_data["page"].object_list.count(), 1)

    def test_pagination(self):
        for i in range(30):
            _make_state(self.cam, self.p1)
        resp1 = self.client.get("/events/?page=1")
        self.assertEqual(resp1.status_code, 200)
        page1 = resp1.context_data["page"]
        self.assertEqual(len(page1.object_list), 25)
        self.assertTrue(page1.has_next())

        resp2 = self.client.get("/events/?page=2")
        self.assertEqual(resp2.status_code, 200)
        page2 = resp2.context_data["page"]
        self.assertEqual(len(page2.object_list), 5)
        self.assertFalse(page2.has_next())


class EventDetailViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(
            name=_test_name("prompt1"), prompt="p", positive_keyword="是",
        )

    def test_found(self):
        s = _make_state(self.cam, self.p, hit=True)
        resp = self.client.get(f"/events/{s.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("prompt1".encode(), resp.content)
        self.assertIn("cam1".encode(), resp.content)

    def test_404(self):
        resp = self.client.get("/events/99999/")
        self.assertEqual(resp.status_code, 404)

    def test_toggle_pause_pauses_prompt(self):
        """B4: 事件详情页 POST /toggle-pause/ → toggle prompt.manual_paused。"""
        s = _make_state(self.cam, self.p, hit=True)
        self.assertFalse(self.p.manual_paused)
        resp = self.client.post(f"/events/{s.id}/toggle-pause/")
        self.assertEqual(resp.status_code, 302)
        self.p.refresh_from_db()
        self.assertTrue(self.p.manual_paused)
        # 第二次 toggle 恢复
        resp2 = self.client.post(f"/events/{s.id}/toggle-pause/")
        self.assertEqual(resp2.status_code, 302)
        self.p.refresh_from_db()
        self.assertFalse(self.p.manual_paused)

    def test_toggle_pause_shows_message(self):
        """C7 兜底：messages.success 真的显示给用户（context_processor.messages 生效）。"""
        s = _make_state(self.cam, self.p, hit=True)
        resp = self.client.post(f"/events/{s.id}/toggle-pause/", follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("已暂停".encode(), resp.content)
        # 第二次恢复
        resp2 = self.client.post(f"/events/{s.id}/toggle-pause/", follow=True)
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("已恢复".encode(), resp2.content)

    def test_toggle_pause_404(self):
        resp = self.client.post("/events/99999/toggle-pause/")
        self.assertEqual(resp.status_code, 404)

    def test_toggle_pause_get_not_allowed(self):
        """GET 不允许（@require_POST）→ 405。"""
        s = _make_state(self.cam, self.p, hit=True)
        resp = self.client.get(f"/events/{s.id}/toggle-pause/")
        self.assertEqual(resp.status_code, 405)


class LogsViewTest(_Base):
    def test_placeholder(self):
        resp = self.client.get("/logs/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("暂未实装".encode(), resp.content)


class ImageUrlAttachmentTest(_Base):
    """event_detail 把 MEDIA_ROOT 内路径转 img{N}_url（template 用）。"""

    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(
            name=_test_name("prompt1"), prompt="p", positive_keyword="是",
        )

    def test_img_inside_media_root_renders_url(self):
        abs_path = str(Path(settings.MEDIA_ROOT) / "frames" / "2026-09-01" / "1_1.jpg")
        s = _make_state(self.cam, self.p, img1=abs_path)
        resp = self.client.get(f"/events/{s.id}/")
        self.assertEqual(resp.status_code, 200)
        expected = settings.MEDIA_URL + "frames/2026-09-01/1_1.jpg"
        self.assertIn(expected.encode(), resp.content)

    def test_img_outside_media_root_no_url(self):
        s = _make_state(self.cam, self.p, img1="D:\\elsewhere\\x.jpg")
        resp = self.client.get(f"/events/{s.id}/")
        self.assertEqual(resp.status_code, 200)
        # 路径不在 MEDIA_ROOT 下 → img1_url 空 → 模板走 else 分支（无 <img>）
        # 校验不应渲染 src 指向 elsewhere 路径
        self.assertNotIn(b"D:\\\\elsewhere", resp.content)
        self.assertNotIn(b"D:/elsewhere", resp.content)


class EventsListFilterKindTest(_Base):
    """B11：kind 过滤（judge / describe）。"""

    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p_judge = VLMPromptConfig.objects.create(
            name=_test_name("judge_p"), prompt="p", positive_keyword="是", kind="judge",
        )
        self.p_describe = VLMPromptConfig.objects.create(
            name=_test_name("describe_p"), prompt="p", positive_keyword="", kind="describe",
        )

    def test_filter_kind_judge(self):
        _make_state(self.cam, self.p_judge)
        _make_state(self.cam, self.p_describe)
        resp = self.client.get("/events/?kind=judge")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        ids = [s.prompt_config_id for s in page.object_list]
        self.assertEqual(ids, [self.p_judge.id])

    def test_filter_kind_describe(self):
        _make_state(self.cam, self.p_judge)
        _make_state(self.cam, self.p_describe)
        resp = self.client.get("/events/?kind=describe")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        ids = [s.prompt_config_id for s in page.object_list]
        self.assertEqual(ids, [self.p_describe.id])

    def test_filter_kind_invalid_ignored(self):
        _make_state(self.cam, self.p_judge)
        _make_state(self.cam, self.p_describe)
        resp = self.client.get("/events/?kind=garbage")
        self.assertEqual(resp.status_code, 200)
        page = resp.context_data["page"]
        self.assertEqual(page.object_list.count(), 2)


class EventsListDateFilterTest(_Base):
    """B9：start_dt / end_dt 过滤（detected_at）。"""

    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(
            name=_test_name("p"), prompt="p", positive_keyword="是",
        )
        # 3 条：today / yesterday / 2 days ago
        now = timezone.now()
        self.s_today = _make_state(self.cam, self.p)
        self.s_yesterday = _make_state(self.cam, self.p)
        self.s_2days = _make_state(self.cam, self.p)
        VLMCheckState.objects.filter(id=self.s_today.id).update(
            detected_at=now - timedelta(hours=1),
        )
        VLMCheckState.objects.filter(id=self.s_yesterday.id).update(
            detected_at=now - timedelta(days=1),
        )
        VLMCheckState.objects.filter(id=self.s_2days.id).update(
            detected_at=now - timedelta(days=2),
        )

    def test_filter_start_dt(self):
        # start_dt = today 中午 → 只 today 那条
        start = (timezone.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        resp = self.client.get(f"/events/?start_dt={start}")
        self.assertEqual(resp.status_code, 200)
        ids = [s.id for s in resp.context_data["page"].object_list]
        self.assertEqual(ids, [self.s_today.id])

    def test_filter_end_dt(self):
        # end_dt = today 早晨（now - 12h）→ 只 yesterday + 2 days ago 那 2 条
        end = (timezone.now() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
        resp = self.client.get(f"/events/?end_dt={end}")
        self.assertEqual(resp.status_code, 200)
        ids = sorted(s.id for s in resp.context_data["page"].object_list)
        self.assertEqual(ids, sorted([self.s_yesterday.id, self.s_2days.id]))

    def test_filter_start_end_dt_range(self):
        # 范围：yesterday ± 6h → 应包含 yesterday，不含 today / 2 days ago
        start = (timezone.now() - timedelta(days=1, hours=6)).strftime("%Y-%m-%dT%H:%M")
        end = (timezone.now() - timedelta(hours=18)).strftime("%Y-%m-%dT%H:%M")
        resp = self.client.get(f"/events/?start_dt={start}&end_dt={end}")
        self.assertEqual(resp.status_code, 200)
        ids = [s.id for s in resp.context_data["page"].object_list]
        self.assertEqual(ids, [self.s_yesterday.id])

    def test_filter_invalid_datetime_ignored(self):
        resp = self.client.get("/events/?start_dt=garbage")
        self.assertEqual(resp.status_code, 200)
        # 非法值忽略 → 全部 3 条
        self.assertEqual(resp.context_data["page"].object_list.count(), 3)

    def test_filter_out_of_range_datetime_ignored(self):
        """C2：parse_datetime 对越界值（'2024-13-45T10:00'）raise ValueError，必须 catch。"""
        resp = self.client.get("/events/?start_dt=2024-13-45T10:00")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context_data["page"].object_list.count(), 3)

    def test_filter_aware_datetime_stripped(self):
        """C3 兜底：parse_datetime 带 'Z' / '+08:00' → aware；strip tz 后再比较。"""
        resp = self.client.get("/events/?start_dt=2020-01-01T00:00:00Z")
        self.assertEqual(resp.status_code, 200)
        # 远早于所有 state → 全 3 条（证明 aware 被正确 strip，没 500）
        self.assertEqual(resp.context_data["page"].object_list.count(), 3)
        resp2 = self.client.get("/events/?start_dt=2099-01-01T00:00:00Z")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.context_data["page"].object_list.count(), 0)

    def test_filter_end_before_start_ignored(self):
        # end_dt < start_dt → 忽略 end_dt，按 start_dt 过滤
        start = (timezone.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        end = (timezone.now() - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M")
        resp = self.client.get(f"/events/?start_dt={start}&end_dt={end}")
        self.assertEqual(resp.status_code, 200)
        ids = [s.id for s in resp.context_data["page"].object_list]
        # 只按 start_dt 过滤（end_dt 被忽略）→ 只 today 那条
        self.assertEqual(ids, [self.s_today.id])


class EventDetailHumanTsTest(_Base):
    """B10：event_detail 把 ts_list 渲染成 "2026年09月01日 13:45:30" 格式。"""

    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(
            name=_test_name("p"), prompt="p", positive_keyword="是",
        )

    def test_event_detail_human_ts_list(self):
        # 1756699200 = 2025-09-01 04:00:00 UTC = 2025-09-01 12:00:00 上海时间
        s = _make_state(self.cam, self.p, ts_list=[1756699200, 1756699201, 1756699202])
        resp = self.client.get(f"/events/{s.id}/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode("utf-8")
        # 上海时区下应为 "2025年09月01日 12:00:00" / "12:00:01" / "12:00:02"
        self.assertIn("2025年09月01日 12:00:00", content)
        self.assertIn("年", content)
        self.assertIn("月", content)
        self.assertIn("日", content)
        self.assertIn(":", content)


class EventDetailChineseSynonymsTest(_Base):
    """B7：event_detail dl 字段加中文同义词（small class="text-muted"）。"""

    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(name=_test_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(
            name=_test_name("p"), prompt="p", positive_keyword="是",
        )

    def test_event_detail_chinese_synonyms(self):
        s = _make_state(self.cam, self.p)
        resp = self.client.get(f"/events/{s.id}/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode("utf-8")
        for cn in (
            "（记录编号）",
            "（检测时间）",
            "（摄像头）",
            "（检查项）",
            "（帧时间戳序列（秒））",
            "（是否命中）",
            "（VLM 回答文本）",
            "（VLM 原始响应）",
            "（窗口内是否有 baby）",
            "（失败原因）",
            "（重试次数）",
            "（是否已发通知）",
            "（用户已标误报）",
            "（已自动静默）",
            "（扩展数据）",
            "（写入时间）",
        ):
            self.assertIn(cn, content, f"missing Chinese synonym: {cn}")

class EventsBulkDeleteTest(_Base):
    """事件批量删除 + 异步 worker 测试。"""

    def setUp(self):
        super().setUp()
        # Django TestCase 在事务里跑，close_old_connections 会让后续 query 失败
        self._conn_patcher = patch("apps.dashboard.bulk_delete._release_db_connection")
        self._conn_patcher.start()
        self.addCleanup(self._conn_patcher.stop)
        self.cam = Camera.objects.create(name=_test_name("cam"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p = VLMPromptConfig.objects.create(name=_test_name("p"), prompt="p", positive_keyword="是")

    def test_requires_post(self):
        resp = self.client.get("/events/bulk-delete/")
        self.assertEqual(resp.status_code, 405)

    def test_post_redirects_with_filters(self):
        _make_state(self.cam, self.p)
        _make_state(self.cam, self.p)
        with patch("apps.dashboard.views._run_bulk_delete_in_thread") as mock_thread:
            resp = self.client.post("/events/bulk-delete/", {"cam_id": str(self.cam.id), "hit": "0"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, f"/events/?cam_id={self.cam.id}&hit=0")
        args, _ = mock_thread.call_args
        self.assertEqual(args[0]["cam_id"], str(self.cam.id))
        self.assertEqual(args[0]["hit"], "0")
        # n_total 不传：worker 自己算（test_filter_count_logged 验证）

    def test_filter_count_logged(self):
        for _ in range(3):
            _make_state(self.cam, self.p)
        with patch("apps.dashboard.views._run_bulk_delete_in_thread") as mock_thread:
            resp = self.client.post("/events/bulk-delete/", {"cam_id": str(self.cam.id)})
        self.assertEqual(resp.status_code, 302)
        # view 拿 n_total 只给 flash 消息用：检查 flash 文本含「3」
        from django.contrib.messages import get_messages
        msgs = [str(m) for m in get_messages(resp.wsgi_request)]
        self.assertTrue(any("3 条" in m for m in msgs))

    def test_filter_dry_run_only_counts(self):
        """view 算 n_total 给 flash 消息：hit=1 filter 应返回 0"""
        _make_state(self.cam, self.p, hit=False)
        with patch("apps.dashboard.views._run_bulk_delete_in_thread"):
            resp = self.client.post("/events/bulk-delete/", {"hit": "1"})
        from django.contrib.messages import get_messages
        msgs = [str(m) for m in get_messages(resp.wsgi_request)]
        self.assertTrue(any("0 条" in m for m in msgs))

    def test_thread_actually_deletes(self):
        _make_state(self.cam, self.p)
        _make_state(self.cam, self.p)
        from apps.dashboard.bulk_delete import _run_bulk_delete
        _run_bulk_delete({"cam_id": "", "prompt_id": "", "kind": "", "hit": "", "start_dt": "", "end_dt": ""})
        self.assertEqual(VLMCheckState.objects.count(), 0)

    def test_keeps_non_matching(self):
        other_cam = Camera.objects.create(name=_test_name("other"), source_type=Camera.SOURCE_FILE, is_active=False)
        s_match = _make_state(self.cam, self.p)
        s_keep = _make_state(other_cam, self.p)
        from apps.dashboard.bulk_delete import _run_bulk_delete
        _run_bulk_delete({"cam_id": str(self.cam.id), "prompt_id": "", "kind": "", "hit": "", "start_dt": "", "end_dt": ""})
        self.assertFalse(VLMCheckState.objects.filter(pk=s_match.pk).exists())
        self.assertTrue(VLMCheckState.objects.filter(pk=s_keep.pk).exists())

    def test_image_removed_when_no_other_reference(self):
        img_path = str(Path(settings.MEDIA_ROOT) / "frames" / "bulk_test_1.jpg")
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
        Path(img_path).write_bytes(b"fake")
        _make_state(self.cam, self.p, img1=img_path)
        from apps.dashboard.bulk_delete import _run_bulk_delete
        _run_bulk_delete({"cam_id": "", "prompt_id": "", "kind": "", "hit": "", "start_dt": "", "end_dt": ""})
        self.assertFalse(Path(img_path).exists())

    def test_image_kept_when_other_reference(self):
        img_path = str(Path(settings.MEDIA_ROOT) / "frames" / "bulk_test_2.jpg")
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
        Path(img_path).write_bytes(b"fake")
        other_cam = Camera.objects.create(name=_test_name("other2"), source_type=Camera.SOURCE_FILE, is_active=False)
        _make_state(self.cam, self.p, img1=img_path)
        _make_state(other_cam, self.p, img1=img_path)
        from apps.dashboard.bulk_delete import _run_bulk_delete
        _run_bulk_delete({"cam_id": str(self.cam.id), "prompt_id": "", "kind": "", "hit": "", "start_dt": "", "end_dt": ""})
        self.assertTrue(Path(img_path).exists())
        Path(img_path).unlink()

    def test_invalid_filter_ignored(self):
        for _ in range(2):
            _make_state(self.cam, self.p)
        with patch("apps.dashboard.views._run_bulk_delete_in_thread"):
            resp = self.client.post("/events/bulk-delete/", {"start_dt": "garbage"})
        from django.contrib.messages import get_messages
        msgs = [str(m) for m in get_messages(resp.wsgi_request)]
        self.assertTrue(any("2 条" in m for m in msgs))

    def test_empty_filter_deletes_all(self):
        for _ in range(3):
            _make_state(self.cam, self.p)
        from apps.dashboard.bulk_delete import _run_bulk_delete
        _run_bulk_delete({"cam_id": "", "prompt_id": "", "kind": "", "hit": "", "start_dt": "", "end_dt": ""})
        self.assertEqual(VLMCheckState.objects.count(), 0)


# ---------------------------------------------------------------------------
# 音频线控制页（Phase 5，spec §9.3）
# ---------------------------------------------------------------------------
class AudioControlViewTest(_Base):
    """音频控制页渲染 + 启停端点（worker 全部 mock，不真起进程）。"""

    PATH = "/audio-control/"

    def _status(self, **overrides) -> dict:
        base = {
            "enabled": True, "desired": "on",
            "running": True, "pid": 4321, "stale": False,
            "heartbeat_ok": True, "heartbeat_age_sec": 1.0,
            "heartbeat_timeout_sec": 10, "cameras": [],
            "python": "python.exe", "pid_file": "x.pid", "log_file": "x.log",
            "uptime_sec": 12.0,
            "audio_desc_configured": False, "audio_desc_url": "",
            "audio_desc_model": "", "audio_desc_provider": "",
        }
        base.update(overrides)
        return base

    def _messages(self, resp) -> list[str]:
        from django.contrib.messages import get_messages
        return [str(m) for m in get_messages(resp.wsgi_request)]

    def test_get_renders_control_page(self):
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(AudioWorkerManager, "status",
                          return_value=self._status()), \
                patch.object(AudioWorkerManager, "tail_log", return_value=""):
            resp = self.client.get(self.PATH)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("音频控制", resp.content.decode())

    def test_get_shows_stale_banner(self):
        """进程活着但心跳停 → 页面必须提示「卡死」。"""
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(
            AudioWorkerManager, "status",
            return_value=self._status(stale=True, heartbeat_ok=False),
        ), patch.object(AudioWorkerManager, "tail_log", return_value=""):
            resp = self.client.get(self.PATH)
        self.assertIn("卡死", resp.content.decode())

    def test_get_survives_status_error(self):
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(AudioWorkerManager, "status",
                          side_effect=RuntimeError("db down")):
            resp = self.client.get(self.PATH)
        self.assertEqual(resp.status_code, 200)

    def test_on_invokes_start_and_redirects(self):
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(AudioWorkerManager, "start",
                          return_value={"started": True, "pid": 99}) as start:
            resp = self.client.post(self.PATH + "on/")
        start.assert_called_once()
        self.assertEqual(resp.status_code, 302)

    def test_on_reports_start_error_to_user(self):
        """WorkerLockError 类失败必须出现在页面上，不能静默。"""
        from apps.audio_detect.worker_manager import (
            AudioWorkerManager,
            AudioWorkerStartError,
        )
        with patch.object(AudioWorkerManager, "start",
                          side_effect=AudioWorkerStartError("已有音频 worker 在运行")):
            resp = self.client.post(self.PATH + "on/")
        self.assertTrue(any("已有音频 worker 在运行" in m for m in self._messages(resp)))

    def test_on_when_already_running_shows_info(self):
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(AudioWorkerManager, "start",
                          return_value={"started": False, "pid": 99,
                                        "note": "already_running"}):
            resp = self.client.post(self.PATH + "on/")
        self.assertTrue(any("已在运行" in m for m in self._messages(resp)))

    def test_off_invokes_stop(self):
        from apps.audio_detect.worker_manager import AudioWorkerManager
        with patch.object(AudioWorkerManager, "stop",
                          return_value=True) as stop:
            resp = self.client.post(self.PATH + "off/")
        stop.assert_called_once()
        self.assertEqual(resp.status_code, 302)

    def test_on_off_require_post(self):
        self.assertEqual(self.client.get(self.PATH + "on/").status_code, 405)
        self.assertEqual(self.client.get(self.PATH + "off/").status_code, 405)
