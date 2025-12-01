from django.urls import path
from . import views

urlpatterns = [
    path('', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('form/', views.input_form, name='form'),
    path('chat/', views.chat_view, name='chat'),
    # Remove or comment out the problematic report line for now
    # path('report/', views.report_fake_profile, name='report'),
    # Admin routes
    path('admin/analytics/', views.admin_analytics, name='admin_analytics'),
    path('admin/review/', views.admin_review_detections, name='admin_review_detections'),
    # New report routes
    path('submit-report/', views.submit_report, name='submit_report'),
    path('reports/', views.reports_list, name='reports_list'),
]