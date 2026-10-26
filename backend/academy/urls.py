"""URL routes for accounts, progress, notebooks and OAuth, mounted under ``/api/``."""

from django.urls import include, path
from oauth2_provider import views as oauth_views

from academy import views

# Only the endpoints JupyterHub's sign-in needs. django-oauth-toolkit's own
# URL module also ships application-management pages that would let any
# signed-in user register OAuth clients; those are deliberately not mounted.
oauth_patterns = (
    [
        path("authorize/", oauth_views.AuthorizationView.as_view(), name="authorize"),
        path("token/", oauth_views.TokenView.as_view(), name="token"),
        path("revoke_token/", oauth_views.RevokeTokenView.as_view(), name="revoke-token"),
    ],
    "oauth2_provider",
)

urlpatterns = [
    path("auth/session/", views.SessionView.as_view(), name="auth-session"),
    path("auth/register/", views.RegisterView.as_view(), name="auth-register"),
    path("auth/login/", views.LoginView.as_view(), name="auth-login"),
    path("auth/logout/", views.LogoutView.as_view(), name="auth-logout"),
    path("auth/hub-user/", views.HubUserView.as_view(), name="auth-hub-user"),
    path("progress/", views.ProgressView.as_view(), name="progress"),
    path("notebooks/launch/", views.NotebookLaunchView.as_view(), name="notebook-launch"),
    path("oauth/", include(oauth_patterns)),
]
