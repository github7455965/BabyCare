"""配置页（Prompt / 摄像头 / 通知目标）视图与表单测试。

测试隔离（重要，改动原因见下）
------------------------------
**全部用 `django.test.TestCase`**：每个用例跑在独立 test 库里，并且结束时自动回滚。

为什么不能用 plain `unittest.TestCase`（踩过的坑）
------------------------------------------------
1. Django 只在"这次测试选择里存在 `django.test.TestCase`"时才创建 test 库；
   若整批测试都是 plain unittest / SimpleTestCase，它会判定"用不到 DB"并
   **跳过 test 库创建**（日志里的 "Skipping setup of unused database(s)"），
   于是测试直接打在**真实库**上，把 `vlm_camera` / `vlm_prompt_config` 越写越脏；
2. 即使 test 库建了，plain unittest **不回滚**，同一次 run 里后面的用例会撞上
   前面用例留下的数据：unique 冲突（200 而不是 302）、列表非空（空状态断言失败）、
   分页错位（新建的行掉到第 2 页）。

所以这里统一成 TestCase。`_Base._purge` 只是**二次保险**：万一有人用
`python -m unittest` 直接跑（既没有 test 库也没有回滚），也不会在真实库留东西。
"""

import os
import unittest
import uuid
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.test import Client, TestCase  # noqa: E402

from apps.streaming.models import Camera  # noqa: E402
from apps.vlm.models import (  # noqa: E402
    NotifyTarget,
    PromptAudioRule,
    PromptNotifyTarget,
    VLMPromptConfig,
)


def _test_name(base: str) -> str:
    """测试 cam/prompt name 加 uuid 后缀，避免撞已有 unique 约束。"""
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


class _Base(TestCase):
    """公共基类：uuid 前缀命名 + 兜底清理。"""

    def setUp(self):
        from django.conf import settings
        self._orig_allowed = settings.ALLOWED_HOSTS
        settings.ALLOWED_HOSTS = list(self._orig_allowed) + ["testserver"]
        self.addCleanup(setattr, settings, "ALLOWED_HOSTS", self._orig_allowed)

        # 本用例新建的行都带这个前缀 → 可以精确回收，不会碰到别人的数据
        self._tok = f"t_{uuid.uuid4().hex[:8]}_"
        self.addCleanup(self._purge)

        self.client = Client()
        patcher = patch("apps.vlm.signals._get_manager", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _name(self, base: str) -> str:
        """本用例独占的行名（避免撞已有唯一约束，也便于回收）。"""
        return f"{self._tok}{base}"

    def _purge(self):
        """删掉本用例建过的行（按 uuid 前缀）。

        TestCase 本身会回滚；这里是二次保险，保证用 `python -m unittest`
        直接跑时也不残留。
        """
        VLMPromptConfig.objects.filter(name__startswith=self._tok).delete()
        NotifyTarget.objects.filter(name__startswith=self._tok).delete()
        Camera.objects.filter(name__startswith=self._tok).delete()


class ListViewTest(_Base):
    def test_empty(self):
        resp = self.client.get("/config/prompts/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"\xe6\x97\xa0", resp.content)

    def test_with_one(self):
        p = VLMPromptConfig.objects.create(name=self._name("口鼻遮盖"), prompt="看口鼻", positive_keyword="是")
        resp = self.client.get("/config/prompts/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(p.name.encode(), resp.content)


class CreateViewTest(_Base):
    def test_get(self):
        resp = self.client.get("/config/prompts/new/")
        self.assertEqual(resp.status_code, 200)

    def test_post_ok(self):
        name = self._name("口鼻遮盖")
        data = {
            "name": name, "kind": "judge",
            "prompt": "请检查婴儿是否被口鼻遮盖", "positive_keyword": "是",
            "window_sec": 10, "weekdays": "1,2,3,4,5,6,7",
            "enabled": "on", "notify_on_hit": "on",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(VLMPromptConfig.objects.filter(name=name).exists())

    def test_post_ok_redirects_to_list(self):
        """B1: 创建成功后跳列表页（不再停在创建页）。"""
        data = {
            "name": self._name("跳列表验证"), "kind": "judge",
            "prompt": "p", "positive_keyword": "是",
            "window_sec": 10, "weekdays": "1,2,3,4,5,6,7",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/config/prompts/")

    def test_post_invalid_weekdays(self):
        data = {
            "name": "X", "kind": "judge", "prompt": "p", "positive_keyword": "是",
            "window_sec": 10, "weekdays": "abc",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 200)


class DetailViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.p = VLMPromptConfig.objects.create(name=self._name("原名"), prompt="p", positive_keyword="是")

    def test_get(self):
        resp = self.client.get(f"/config/prompts/{self.p.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"\xe5\x8e\x9f\xe5\x90\x8d", resp.content)
        # C4：B4 把 manual_paused 搬到事件详情页，模板不应再渲染该字段
        self.assertNotIn(b"manual_paused", resp.content)

    def test_post_update(self):
        new_name = self._name("改名")
        resp = self.client.post(f"/config/prompts/{self.p.pk}/", {
            "name": new_name, "kind": "describe", "prompt": "p2", "positive_keyword": "no",
            "window_sec": 20, "weekdays": "1,2,3",
        })
        self.assertEqual(resp.status_code, 302)
        self.p.refresh_from_db()
        self.assertEqual(self.p.name, new_name)
        self.assertEqual(self.p.kind, "describe")
        self.assertEqual(self.p.window_sec, 20)


class DeleteViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.p = VLMPromptConfig.objects.create(name=self._name("待删"), prompt="p", positive_keyword="是")

    def test_post_deletes(self):
        resp = self.client.post(f"/config/prompts/{self.p.pk}/delete/")
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(VLMPromptConfig.objects.filter(pk=self.p.pk).exists())

    def test_get_405(self):
        resp = self.client.get(f"/config/prompts/{self.p.pk}/delete/")
        self.assertEqual(resp.status_code, 405)


class WeekdaysCleanTest(TestCase):
    """纯表单校验；用 TestCase 是为了让它也算"需要 DB"，

    避免整批测试选择里没有 TestCase 时 Django 跳过 test 库创建。
    """

    _BASE = {"name": "x", "prompt": "p", "positive_keyword": "是",
             "kind": "judge", "window_sec": 10, "weekdays": ""}

    def _is_valid(self, value):
        from apps.config_panel.forms import VLMPromptConfigForm
        f = VLMPromptConfigForm(data={**self._BASE, "weekdays": value})
        return f.is_valid()

    def test_empty_default(self):
        from apps.config_panel.forms import VLMPromptConfigForm
        f = VLMPromptConfigForm(data=self._BASE)
        self.assertTrue(f.is_valid())
        self.assertEqual(f.cleaned_data["weekdays"], "1,2,3,4,5,6,7")

    def test_valid(self):
        from apps.config_panel.forms import VLMPromptConfigForm
        f = VLMPromptConfigForm(data={**self._BASE, "weekdays": "1,2,3"})
        self.assertTrue(f.is_valid())
        self.assertEqual(f.cleaned_data["weekdays"], "1,2,3")

    def test_invalid_out_of_range(self):
        self.assertFalse(self._is_valid("8"))

    def test_invalid_alpha(self):
        self.assertFalse(self._is_valid("abc"))


class PromptTargetWarningTest(TestCase):
    """目标类别关键字检查：**只警告、不拦保存**（2026-09-14 改）。

    原实现走 ``add_error`` → 用户写不出自己需要的文案（多选时文案缺某类别关键字就
    保存不了）。现在降级为 ``form.warnings`` + ``messages.warning``：保存照常成功，
    提示仍能看到。
    """

    _BASE = {"name": "多选缺词", "prompt": "图中有大人吗？", "positive_keyword": "是",
             "kind": "judge", "window_sec": 10, "weekdays": ""}

    def _form(self, **overrides):
        from apps.config_panel.forms import VLMPromptConfigForm

        return VLMPromptConfigForm(data={**self._BASE, **overrides})

    def test_missing_keyword_is_warning_not_error(self):
        """勾 baby+person、文案只提"大人" → 仍 is_valid()，只在 warnings 里提示。"""
        form = self._form(target_classes=["baby", "person"])
        self.assertTrue(form.is_valid())
        self.assertFalse(form.errors)          # 不再挂在 prompt 字段上（保存不会被拦）
        self.assertEqual(len(form.warnings), 1)
        self.assertIn("baby", form.warnings[0])

    def test_all_mentioned_no_warning(self):
        form = self._form(
            prompt="图中是否有宝宝（婴儿/小孩）？是否有大人？",
            target_classes=["baby", "person"],
        )
        self.assertTrue(form.is_valid())
        self.assertEqual(form.warnings, [])

    def test_single_target_is_not_checked(self):
        """单选不校验：文案完全没提 baby 也不警告。"""
        form = self._form(prompt="描述画面内容。", target_classes=["baby"])
        self.assertTrue(form.is_valid())
        self.assertEqual(form.warnings, [])

    def test_existing_config_warns_on_page_load(self):
        """编辑已有配置：进页面（未绑定表单）就提示，不必先提交一次。"""
        from apps.config_panel.forms import VLMPromptConfigForm

        p = VLMPromptConfig.objects.create(
            name="已有配置", prompt="图中有大人吗？", positive_keyword="是",
            target_classes="baby,person",
        )
        self.assertEqual(len(VLMPromptConfigForm(instance=p).warnings), 1)

    def test_post_saves_and_leaves_warning_message(self):
        """POST 保存成功（302）并留下 warning message。"""
        from django.contrib.messages import get_messages

        p = VLMPromptConfig.objects.create(
            name="保存测试", prompt="图中有大人吗？", positive_keyword="是",
            target_classes="baby,person",
        )
        resp = self.client.post(
            f"/config/prompts/{p.pk}/",
            {**self._BASE, "name": p.name, "target_classes": ["baby", "person"]},
        )
        self.assertEqual(resp.status_code, 302)
        p.refresh_from_db()
        self.assertEqual(p.target_classes, "baby,person")

        msgs = list(get_messages(resp.wsgi_request))
        self.assertTrue(any("baby" in m.message for m in msgs))
        self.assertTrue(any(m.level_tag == "warning" for m in msgs))


class CameraMultiSelectTest(_Base):
    def setUp(self):
        super().setUp()
        self.cam_a = Camera.objects.create(name=self._name("cam_a"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.cam_b = Camera.objects.create(name=self._name("cam_b"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.cam_c = Camera.objects.create(name=self._name("cam_c"), source_type=Camera.SOURCE_FILE, is_active=False)

    def test_multi_select(self):
        name = self._name("多选测试")
        data = {
            "name": name,
            "kind": "judge",
            "prompt": "p",
            "positive_keyword": "是",
            "camera_ids": [self.cam_a.id, self.cam_c.id],
            "window_sec": 10,
            "weekdays": "1,2,3,4,5,6,7",
            "enabled": "on",
            "notify_on_hit": "on",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 302)

        p = VLMPromptConfig.objects.get(name=name)
        linked_ids = list(p.camera_ids.values_list("id", flat=True))
        self.assertEqual(len(linked_ids), 2)
        self.assertIn(self.cam_a.id, linked_ids)
        self.assertIn(self.cam_c.id, linked_ids)
        self.assertNotIn(self.cam_b.id, linked_ids)

        list_resp = self.client.get("/config/prompts/")
        self.assertEqual(list_resp.status_code, 200)
        self.assertIn(b"2", list_resp.content)


class DetailNotFoundTest(_Base):
    def test_all_routes_404(self):
        self.assertEqual(self.client.get("/config/prompts/99999/").status_code, 404)
        self.assertEqual(self.client.post("/config/prompts/99999/").status_code, 404)
        self.assertEqual(self.client.post("/config/prompts/99999/delete/").status_code, 404)


class DuplicateNameTest(_Base):
    def test_unique_violation_renders_form(self):
        dup = self._name("冲突名")
        VLMPromptConfig.objects.create(name=dup, prompt="p", positive_keyword="是")

        data = {
            "name": dup,
            "kind": "judge",
            "prompt": "p2",
            "positive_keyword": "是",
            "window_sec": 10,
            "weekdays": "1,2,3,4,5,6,7",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 200)

        ctx = resp.context
        if isinstance(ctx, list):
            ctx = ctx[0] if ctx else {}
        form = ctx.get("form") if hasattr(ctx, "get") else None
        form_name_errors = form.errors.get("name") if form and hasattr(form, "errors") else None

        content = resp.content.decode("utf-8", errors="ignore")
        content_hit = any(kw in content for kw in ("已存在", "存在", "unique", "UNIQUE"))

        self.assertTrue(
            bool(form_name_errors) or content_hit,
            "expected either form 'name' errors or content uniqueness marker",
        )

        self.assertEqual(VLMPromptConfig.objects.filter(name=dup).count(), 1)


class ManualPausedSignalTest(_Base):
    def setUp(self):
        super().setUp()
        self.mock_mgr = MagicMock()
        patcher = patch("apps.vlm.signals._get_manager", lambda: self.mock_mgr)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_then_remove_via_signal(self):
        p = VLMPromptConfig.objects.create(
            name=self._name("跑"),
            prompt="p",
            positive_keyword="是",
            enabled=True,
            manual_paused=False,
        )
        self.assertTrue(VLMPromptConfig.objects.filter(pk=p.pk).exists())

        p.manual_paused = True
        p.save()

        self.mock_mgr.add_prompt.assert_called_with(p.pk)
        self.mock_mgr.remove_prompt.assert_called_with(p.pk)


class CameraListViewTest(_Base):
    def setUp(self):
        super().setUp()

    def test_empty(self):
        resp = self.client.get("/config/cameras/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("新建摄像头".encode(), resp.content)

    def test_with_one(self):
        cam = Camera.objects.create(
            name=self._name("卧室"), source_type=Camera.SOURCE_FILE, is_active=False,
            file_path="D:\\video.mp4",
        )
        resp = self.client.get("/config/cameras/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(cam.name.encode(), resp.content)


class CameraCreateViewTest(_Base):
    def setUp(self):
        super().setUp()

    def test_get(self):
        resp = self.client.get("/config/cameras/new/")
        self.assertEqual(resp.status_code, 200)

    def test_post_file_missing_path(self):
        name = self._name("缺路径")
        data = {
            "name": name,
            "source_type": Camera.SOURCE_FILE,
            "file_path": "",
            "onvif_port": 80,
        }
        resp = self.client.post("/config/cameras/new/", data)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Camera.objects.filter(name=name).exists())

    def test_post_onvif_missing_host(self):
        name = self._name("缺host")
        data = {
            "name": name,
            "source_type": Camera.SOURCE_ONVIF,
            "file_path": "",
            "onvif_host": "",
            "onvif_username": "admin",
            "onvif_port": 80,
        }
        resp = self.client.post("/config/cameras/new/", data)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Camera.objects.filter(name=name).exists())

    def test_post_file_ok(self):
        before = Camera.objects.count()
        data = {
            "name": self._name("卧室"),
            "source_type": Camera.SOURCE_FILE,
            "file_path": "D:\\video.mp4",
            "onvif_port": 80,
        }
        resp = self.client.post("/config/cameras/new/", data)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Camera.objects.count(), before + 1)

    def test_post_ok_redirects_to_list(self):
        """B2: 创建成功后跳列表页（不再停在创建页）。"""
        data = {
            "name": self._name("跳列表验证B2"),
            "source_type": Camera.SOURCE_FILE,
            "file_path": "D:\\video.mp4",
            "onvif_port": 80,
        }
        resp = self.client.post("/config/cameras/new/", data)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/config/cameras/")


class CameraDetailViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(
            name=self._name("卧室"), source_type=Camera.SOURCE_FILE, is_active=False,
            file_path="D:\\video.mp4",
        )

    def test_get(self):
        resp = self.client.get(f"/config/cameras/{self.cam.pk}/")
        self.assertEqual(resp.status_code, 200)

    def test_post_update(self):
        new_name = self._name("改名")
        data = {
            "name": new_name,
            "source_type": Camera.SOURCE_FILE,
            "file_path": "D:\\video2.mp4",
            "onvif_port": 80,
        }
        resp = self.client.post(f"/config/cameras/{self.cam.pk}/", data)
        self.assertEqual(resp.status_code, 302)
        self.cam.refresh_from_db()
        self.assertEqual(self.cam.name, new_name)
        self.assertEqual(self.cam.file_path, "D:\\video2.mp4")

    def test_referenced_by(self):
        prompt = VLMPromptConfig.objects.create(
            name=self._name("P1"), prompt="p", positive_keyword="是",
        )
        prompt.camera_ids.add(self.cam)
        resp = self.client.get(f"/config/cameras/{self.cam.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<strong>1</strong>", resp.content)


class CameraDeleteViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.cam = Camera.objects.create(
            name=self._name("待删"), source_type=Camera.SOURCE_FILE, is_active=False,
            file_path="D:\\video.mp4",
        )

    def test_post_deletes(self):
        resp = self.client.post(f"/config/cameras/{self.cam.pk}/delete/")
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(Camera.objects.filter(pk=self.cam.pk).exists())

    def test_get_405(self):
        resp = self.client.get(f"/config/cameras/{self.cam.pk}/delete/")
        self.assertEqual(resp.status_code, 405)


class NotifyTargetListViewTest(_Base):
    def test_empty(self):
        resp = self.client.get("/config/notify-targets/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"\xe6\x9a\x82\xe6\x97\xa0", resp.content)

    def test_with_one(self):
        NotifyTarget.objects.create(name=self._name("dad_phone"), kind="mobile_app", target_id="mobile_app_dad")
        resp = self.client.get("/config/notify-targets/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"dad_phone", resp.content)


class NotifyTargetCreateViewTest(_Base):
    def test_get(self):
        resp = self.client.get("/config/notify-targets/new/")
        self.assertEqual(resp.status_code, 200)

    def test_post_ok_redirects_to_list(self):
        name = self._name("kitchen_speaker")
        resp = self.client.post("/config/notify-targets/new/", {
            "name": name,
            "kind": "speaker",
            "target_id": f"text.{name}",
            "enabled": "on",
        })
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/config/notify-targets/")
        t = NotifyTarget.objects.get(name=name)
        self.assertEqual(t.kind, "speaker")
        self.assertTrue(t.enabled)

    def test_post_invalid(self):
        before = NotifyTarget.objects.count()
        resp = self.client.post("/config/notify-targets/new/", {
            "name": "",
            "kind": "mobile_app",
            "target_id": "",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(NotifyTarget.objects.count(), before)


class NotifyTargetDetailViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.t = NotifyTarget.objects.create(
            name=self._name("dad_phone"), kind="mobile_app", target_id="mobile_app_dad",
        )

    def test_get(self):
        resp = self.client.get(f"/config/notify-targets/{self.t.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"dad_phone", resp.content)

    def test_post_update(self):
        new_name = self._name("mom_phone")
        resp = self.client.post(f"/config/notify-targets/{self.t.pk}/", {
            "name": new_name,
            "kind": "mobile_app",
            "target_id": "mobile_app_mom",
            "enabled": "on",
        })
        self.assertEqual(resp.status_code, 302)
        self.t.refresh_from_db()
        self.assertEqual(self.t.name, new_name)
        self.assertEqual(self.t.target_id, "mobile_app_mom")

    def test_referenced_by(self):
        p = VLMPromptConfig.objects.create(name=self._name("p"), prompt="x", positive_keyword="是")
        p.notify_targets.add(self.t)
        resp = self.client.get(f"/config/notify-targets/{self.t.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"\xe8\xa2\xab Prompt \xe5\xbc\x95\xe7\x94\xa8", resp.content)
        self.assertIn(p.name.encode(), resp.content)


class NotifyTargetDeleteViewTest(_Base):
    def setUp(self):
        super().setUp()
        self.t = NotifyTarget.objects.create(
            name=self._name("dad_phone"), kind="mobile_app", target_id="mobile_app_dad",
        )

    def test_post_deletes(self):
        resp = self.client.post(f"/config/notify-targets/{self.t.pk}/delete/")
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(NotifyTarget.objects.filter(pk=self.t.pk).exists())

    def test_get_405(self):
        resp = self.client.get(f"/config/notify-targets/{self.t.pk}/delete/")
        self.assertEqual(resp.status_code, 405)


class PromptNotifyTargetConditionTest(_Base):
    """Phase 6：每个目标一行「☑ 目标名 + 条件下拉」（spec §5.6）。"""

    def setUp(self):
        super().setUp()
        self.t1 = NotifyTarget.objects.create(
            name=self._name("phone"), kind="mobile_app", target_id="notify.mobile_app_x",
        )
        self.t2 = NotifyTarget.objects.create(
            name=self._name("speaker"), kind="speaker", target_id="text.xiaomi_x",
        )

    def _base_data(self, name):
        return {
            "name": name, "kind": "judge", "prompt": "p", "positive_keyword": "是",
            "window_sec": 10, "weekdays": "1,2,3,4,5,6,7",
            "enabled": "on", "notify_on_hit": "on",
        }

    def test_create_with_per_target_condition(self):
        name = self._name("带条件")
        data = {
            **self._base_data(name),
            f"nt_{self.t1.pk}_enabled": "on",
            f"nt_{self.t1.pk}_condition": "audio_rule",
            f"nt_{self.t2.pk}_enabled": "on",
            f"nt_{self.t2.pk}_condition": "always",
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 302)

        p = VLMPromptConfig.objects.get(name=name)
        links = {l.notify_target_id: l.condition for l in
                 PromptNotifyTarget.objects.filter(prompt_config=p)}
        self.assertEqual(links[self.t1.pk], "audio_rule")
        self.assertEqual(links[self.t2.pk], "always")

    def test_unchecked_target_removed(self):
        name = self._name("取消勾选")
        p = VLMPromptConfig.objects.create(
            name=name, prompt="p", positive_keyword="是",
        )
        PromptNotifyTarget.objects.create(
            prompt_config=p, notify_target=self.t1, condition="always",
        )
        data = self._base_data(name)  # 两个都不勾
        resp = self.client.post(f"/config/prompts/{p.pk}/", data)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(
            PromptNotifyTarget.objects.filter(prompt_config=p).count(), 0,
        )

    def test_condition_updated(self):
        name = self._name("改条件")
        p = VLMPromptConfig.objects.create(
            name=name, prompt="p", positive_keyword="是",
        )
        PromptNotifyTarget.objects.create(
            prompt_config=p, notify_target=self.t1, condition="always",
        )
        data = {
            **self._base_data(name),
            f"nt_{self.t1.pk}_enabled": "on",
            f"nt_{self.t1.pk}_condition": "audio_rule",
        }
        resp = self.client.post(f"/config/prompts/{p.pk}/", data)
        self.assertEqual(resp.status_code, 302)
        link = PromptNotifyTarget.objects.get(prompt_config=p, notify_target=self.t1)
        self.assertEqual(link.condition, "audio_rule")

    def test_detail_renders_rows_and_condition_select(self):
        name = self._name("渲染")
        p = VLMPromptConfig.objects.create(
            name=name, prompt="p", positive_keyword="是",
        )
        PromptNotifyTarget.objects.create(
            prompt_config=p, notify_target=self.t1, condition="audio_rule",
        )
        resp = self.client.get(f"/config/prompts/{p.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(self.t1.name.encode(), resp.content)
        self.assertIn(b"audio_rule", resp.content)


class PromptAudioRuleFormTest(_Base):
    """Phase 6：声音条件随 Prompt 表单保存（spec §5.5）。"""

    def _base_data(self, name):
        return {
            "name": name, "kind": "judge", "prompt": "p", "positive_keyword": "是",
            "window_sec": 10, "weekdays": "1,2,3,4,5,6,7",
            "enabled": "on", "notify_on_hit": "on",
        }

    def test_create_audio_rule(self):
        name = self._name("声音条件")
        data = {
            **self._base_data(name),
            "audio_rule_enabled": "on",
            "audio_rule_condition": "count_cry",
            "audio_rule_window_sec": 120,
            "audio_rule_min_event_count": 2,
        }
        resp = self.client.post("/config/prompts/new/", data)
        self.assertEqual(resp.status_code, 302)

        p = VLMPromptConfig.objects.get(name=name)
        rule = PromptAudioRule.objects.get(prompt_config=p)
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.condition, "count_cry")
        self.assertEqual(rule.window_sec, 120)
        self.assertEqual(rule.min_event_count, 2)

    def test_no_rule_row_when_never_enabled(self):
        """未启用且从未配过 → 不建行（保持 UNKNOWN=no_rule）。"""
        name = self._name("没配声音")
        resp = self.client.post("/config/prompts/new/", self._base_data(name))
        self.assertEqual(resp.status_code, 302)
        p = VLMPromptConfig.objects.get(name=name)
        self.assertFalse(PromptAudioRule.objects.filter(prompt_config=p).exists())

    def test_disable_keeps_row_but_marks_disabled(self):
        name = self._name("关声音")
        p = VLMPromptConfig.objects.create(
            name=name, prompt="p", positive_keyword="是",
        )
        PromptAudioRule.objects.create(
            prompt_config=p, enabled=True, condition="cry", window_sec=60,
        )
        resp = self.client.post(f"/config/prompts/{p.pk}/", self._base_data(name))
        self.assertEqual(resp.status_code, 302)
        rule = PromptAudioRule.objects.get(prompt_config=p)
        self.assertFalse(rule.enabled)
        self.assertEqual(rule.condition, "cry")  # 配置保留

    def test_form_initial_reflects_existing_rule(self):
        from apps.config_panel.forms import VLMPromptConfigForm

        name = self._name("回填")
        p = VLMPromptConfig.objects.create(
            name=name, prompt="p", positive_keyword="是",
        )
        PromptAudioRule.objects.create(
            prompt_config=p, enabled=True, condition="no_speech",
            window_sec=90, min_event_count=3,
        )
        form = VLMPromptConfigForm(instance=p)
        self.assertEqual(form.fields["audio_rule_condition"].initial, "no_speech")
        self.assertEqual(form.fields["audio_rule_window_sec"].initial, 90)
        self.assertEqual(form.fields["audio_rule_min_event_count"].initial, 3)
