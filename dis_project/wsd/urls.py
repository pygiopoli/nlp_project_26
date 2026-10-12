from django.urls import path

from wsd import views

app_name = "wsd"

urlpatterns = [
    path("", views.index, name="index"),
]
