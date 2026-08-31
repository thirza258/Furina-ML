"""URL routes for the Furina ML API, all mounted under ``/api/``."""

from django.urls import path

from core import views

app_name = "core"

urlpatterns = [
    path("health/", views.HealthView.as_view(), name="health"),
    path("assistant/", views.AssistantView.as_view(), name="assistant"),
    path("datasets/upload/", views.DatasetUploadView.as_view(), name="dataset-upload"),
    path("datasets/", views.DatasetListView.as_view(), name="dataset-list"),
    path("datasets/<uuid:pk>/", views.DatasetDetailView.as_view(), name="dataset-detail"),
    path(
        "datasets/<uuid:pk>/download/",
        views.DatasetDownloadView.as_view(),
        name="dataset-download",
    ),
    path(
        "datasets/<uuid:pk>/preprocess/",
        views.DatasetPreprocessView.as_view(),
        name="dataset-preprocess",
    ),
    path("algorithms/", views.AlgorithmListView.as_view(), name="algorithm-list"),
    path("train/", views.TrainView.as_view(), name="train"),
    path("runs/", views.RunListView.as_view(), name="run-list"),
    path("runs/<uuid:pk>/", views.RunDetailView.as_view(), name="run-detail"),
    path("runs/<uuid:pk>/predict/", views.RunPredictView.as_view(), name="run-predict"),
    path(
        "runs/<uuid:pk>/download_model/",
        views.RunDownloadModelView.as_view(),
        name="run-download-model",
    ),
    path("runs/<uuid:pk>/report/", views.RunReportView.as_view(), name="run-report"),
]
