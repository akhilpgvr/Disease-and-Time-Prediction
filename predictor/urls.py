from django.urls import path
from .views import predict_prognosis

urlpatterns = [
    path('predict-prognosis/', predict_prognosis, name='predict-prognosis'),  # Correct URL pattern
]
