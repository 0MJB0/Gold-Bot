from django.contrib import admin
from django.urls import path

from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('news/', news),
    path('predict/', predict),
    path('update-chart/', update_chart, name='update-chart'), 
]   