"""URL routes for the lifecycle API, mounted under ``/api/lifecycle/``."""

from django.urls import path

from lifecycle import views

app_name = "lifecycle"

urlpatterns = [
    # dataset versions -----------------------------------------------------
    path(
        "datasets/<uuid:dataset_id>/versions/",
        views.DatasetVersionListView.as_view(),
        name="version-list",
    ),
    path(
        "versions/<uuid:pk>/",
        views.DatasetVersionDetailView.as_view(),
        name="version-detail",
    ),
    path(
        "versions/<uuid:pk>/lineage/",
        views.DatasetVersionLineageView.as_view(),
        name="version-lineage",
    ),
    path(
        "versions/<uuid:pk>/download/",
        views.DatasetVersionDownloadView.as_view(),
        name="version-download",
    ),
    path(
        "versions/<uuid:pk>/leakage/",
        views.DatasetVersionLeakageView.as_view(),
        name="version-leakage",
    ),
    path(
        "versions/<uuid:pk>/resources/",
        views.DatasetVersionResourcesView.as_view(),
        name="version-resources",
    ),

    # experiments ------------------------------------------------------------
    path("experiments/", views.ExperimentListView.as_view(), name="experiment-list"),
    path("experiments/<uuid:pk>/", views.ExperimentDetailView.as_view(), name="experiment-detail"),
    path("experiments/<uuid:pk>/runs/", views.ExperimentRunsView.as_view(), name="experiment-runs"),
    path("experiments/<uuid:pk>/train/", views.ExperimentTrainView.as_view(), name="experiment-train"),
    path("experiments/<uuid:pk>/search/", views.ExperimentHpoView.as_view(), name="experiment-search"),
    path("experiments/<uuid:pk>/automl/", views.ExperimentAutoMlView.as_view(), name="experiment-automl"),

    # runs ---------------------------------------------------------------------
    path("runs/<uuid:pk>/", views.RunDetailView.as_view(), name="run-detail"),
    path("runs/compare/", views.RunCompareView.as_view(), name="run-compare"),
    path("runs/<uuid:pk>/download_model/", views.RunDownloadModelView.as_view(), name="run-download-model"),
    path("runs/<uuid:pk>/register/", views.RunRegisterModelView.as_view(), name="run-register"),
    path("runs/<uuid:pk>/reproduce/", views.RunReproduceView.as_view(), name="run-reproduce"),
    path("runs/<uuid:pk>/notebook/", views.RunNotebookView.as_view(), name="run-notebook"),

    # jobs ---------------------------------------------------------------------
    path("jobs/<uuid:pk>/", views.JobDetailView.as_view(), name="job-detail"),
    path("jobs/<uuid:pk>/cancel/", views.JobCancelView.as_view(), name="job-cancel"),

    # model registry -------------------------------------------------------------
    path("models/", views.ModelListView.as_view(), name="model-list"),
    path("models/<uuid:pk>/", views.ModelDetailView.as_view(), name="model-detail"),
    path(
        "models/<uuid:pk>/versions/<int:version>/",
        views.ModelVersionDetailView.as_view(),
        name="model-version-detail",
    ),
    path(
        "models/<uuid:pk>/versions/<int:version>/promote/",
        views.ModelVersionPromoteView.as_view(),
        name="model-version-promote",
    ),
    path(
        "models/<uuid:pk>/versions/<int:version>/deploy/",
        views.ModelVersionDeployView.as_view(),
        name="model-version-deploy",
    ),

    # deployments -------------------------------------------------------------------
    path("deployments/", views.DeploymentListView.as_view(), name="deployment-list"),
    path("deployments/<str:slug>/", views.DeploymentDetailView.as_view(), name="deployment-detail"),
    path(
        "deployments/<str:slug>/predict/",
        views.DeploymentPredictView.as_view(),
        name="deployment-predict",
    ),
    path(
        "deployments/<str:slug>/batch-predict/",
        views.DeploymentBatchPredictView.as_view(),
        name="deployment-batch-predict",
    ),
    path(
        "deployments/<str:slug>/openapi/",
        views.DeploymentOpenApiView.as_view(),
        name="deployment-openapi",
    ),
    path(
        "deployments/<str:slug>/snippets/",
        views.DeploymentSnippetsView.as_view(),
        name="deployment-snippets",
    ),
    path(
        "deployments/<str:slug>/monitoring/",
        views.DeploymentMonitoringView.as_view(),
        name="deployment-monitoring",
    ),
    path(
        "deployments/<str:slug>/drift-check/",
        views.DeploymentDriftCheckView.as_view(),
        name="deployment-drift-check",
    ),

    # monitoring -----------------------------------------------------------------------
    path("monitoring/", views.MonitoringOverviewView.as_view(), name="monitoring-overview"),
]
