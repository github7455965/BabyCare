from django.urls import path

from . import audio_views, views

app_name = "dashboard"

urlpatterns = [
    path("events/", views.events_list, name="events_list"),
    path("events/bulk-delete/", views.bulk_delete, name="events_bulk_delete"),
    path("events/<int:pk>/", views.event_detail, name="event_detail"),
    path("events/<int:pk>/toggle-pause/", views.toggle_pause, name="toggle_pause"),
    path("logs/", views.logs, name="logs"),
    path("vlm-control/", views.vlm_control, name="vlm_control"),
    path("vlm-control/llama/off/", views.vlm_llama_off, name="vlm_llama_off"),
    path("vlm-control/llama/on/", views.vlm_llama_on, name="vlm_llama_on"),
    path("audio-control/", views.audio_control, name="audio_control"),
    path("audio-control/on/", views.audio_control_on, name="audio_control_on"),
    path("audio-control/off/", views.audio_control_off, name="audio_control_off"),
    # Phase 7：音频线页面（spec §8）
    path(
        "sound-detection/logs/",
        audio_views.sound_detection_logs,
        name="sound_detection_logs",
    ),
    # 同一份数据的另一种粒度：按音频事件（与按窗口互为切换视图）
    path(
        "sound-detection/events/",
        audio_views.sound_events_list,
        name="sound_events_list",
    ),
    path(
        "sound-detection/cleanup/",
        audio_views.audio_cleanup,
        name="audio_cleanup",
    ),
    path(
        "sound-detection/cleanup/run/",
        audio_views.audio_cleanup_run,
        name="audio_cleanup_run",
    ),
    path(
        "sound-detection/cleanup/orphans/",
        audio_views.audio_cleanup_orphans,
        name="audio_cleanup_orphans",
    ),
    path("sound-events/<int:pk>/", audio_views.audio_event_detail, name="audio_event_detail"),
    path(
        "sound-events/<int:pk>/audio/",
        audio_views.audio_event_file,
        name="audio_event_file",
    ),
    path(
        "sound-events/<int:pk>/segments/<int:seq>/audio/",
        audio_views.audio_segment_file,
        name="audio_segment_file",
    ),
]