"""
Step 8 signals：监听 Camera / VLMPromptConfig 变更，触发 RunnerManager 增删。

监听规则
--------
- Camera post_save（is_active=True 新建/激活）→ add_camera(cam_id)
- Camera post_save（is_active 切 False）→ remove_camera(cam_id)
- Camera post_delete → remove_camera(cam_id)
- VLMPromptConfig post_save（enabled=True / manual_paused=False）→ add_prompt(prompt_id)
- VLMPromptConfig post_save（enabled=False 或 manual_paused=True）→ remove_prompt(prompt_id)
- VLMPromptConfig post_delete → remove_prompt(prompt_id)

注意
----
- signal 在事务 commit 后触发（post_save/post_delete 默认就是），避免读未提交数据
- signal handler 异常不能让主流程挂掉 → 用 logger.exception 兜底
- RunnerManager 没启动时（管理命令 / 测试）直接 return
"""

from __future__ import annotations

import logging

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver


logger = logging.getLogger(__name__)


def _safe(fn, *args, **kwargs) -> None:
    """包一层 try/except；signal handler 不让主流程挂。"""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.exception("[vlm-signal] handler failed: %s", e)


def _get_manager():
    """懒加载 RunnerManager；未启动时返回 None。"""
    try:
        from apps.vlm.runner import PromptRunnerManager

        mgr = PromptRunnerManager.instance()
        if not mgr._started:  # noqa: SLF001  测试 / manage command 路径
            return None
        return mgr
    except Exception as e:  # noqa: BLE001
        logger.warning("[vlm-signal] get manager failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Camera 信号
# ---------------------------------------------------------------------------
@receiver(post_save, sender="streaming.Camera")
def _on_camera_saved(sender, instance, created, **kwargs):
    mgr = _get_manager()
    if mgr is None:
        return
    if instance.is_active:
        _safe(mgr.add_camera, instance.id)
    else:
        _safe(mgr.remove_camera, instance.id)


@receiver(post_delete, sender="streaming.Camera")
def _on_camera_deleted(sender, instance, **kwargs):
    mgr = _get_manager()
    if mgr is None:
        return
    _safe(mgr.remove_camera, instance.id)


# ---------------------------------------------------------------------------
# VLMPromptConfig 信号
# ---------------------------------------------------------------------------
def _is_prompt_runnable(instance) -> bool:
    """prompt 是否应该被 Runner 跑：enabled 且 未暂停。"""
    return bool(instance.enabled and not instance.manual_paused)


@receiver(post_save, sender="vlm.VLMPromptConfig")
def _on_prompt_saved(sender, instance, created, **kwargs):
    mgr = _get_manager()
    if mgr is None:
        return
    if _is_prompt_runnable(instance):
        _safe(mgr.add_prompt, instance.id)
    else:
        _safe(mgr.remove_prompt, instance.id)


@receiver(post_delete, sender="vlm.VLMPromptConfig")
def _on_prompt_deleted(sender, instance, **kwargs):
    mgr = _get_manager()
    if mgr is None:
        return
    _safe(mgr.remove_prompt, instance.id)
