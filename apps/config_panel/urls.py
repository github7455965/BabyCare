from django.urls import path

from . import camera_views, notify_target_views, views

app_name = "config_panel"

urlpatterns = [
    path("prompts/", views.list_view, name="prompts_list"),
    path("prompts/new/", views.create_view, name="prompt_create"),
    path("prompts/<int:pk>/", views.detail_view, name="prompt_detail"),
    path("prompts/<int:pk>/delete/", views.delete_view, name="prompt_delete"),
    path("cameras/", camera_views.list_view, name="camera_list"),
    path("cameras/new/", camera_views.create_view, name="camera_create"),
    path("cameras/<int:pk>/", camera_views.detail_view, name="camera_detail"),
    path("cameras/<int:pk>/delete/", camera_views.delete_view, name="camera_delete"),
    path("notify-targets/", notify_target_views.list_view, name="notify_targets_list"),
    path("notify-targets/new/", notify_target_views.create_view, name="notify_target_create"),
    path("notify-targets/<int:pk>/", notify_target_views.detail_view, name="notify_target_detail"),
    path("notify-targets/<int:pk>/delete/", notify_target_views.delete_view, name="notify_target_delete"),
]