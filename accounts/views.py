# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden
from django.conf import settings

from .forms import ProfileForm, UserRegistrationForm, UserReportForm
from .models import ProfileData, Report, Detection, UserReport
from .blockchain import blockchain

import numpy as np
import pickle
import matplotlib
import base64
from io import BytesIO
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json

# Use Agg backend for matplotlib for headless servers
matplotlib.use('Agg')

# ========== BASIC AUTHENTICATION VIEWS ==========

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            
            # Check if username already exists
            if User.objects.filter(username=username).exists():
                return render(request, 'register.html', {
                    'form': form,
                    'error': 'Username already exists. Please choose a different one.'
                })
            
            # Check if email already exists
            if User.objects.filter(email=email).exists():
                return render(request, 'register.html', {
                    'form': form,
                    'error': 'Email already registered. Please use a different email.'
                })
            
            # Create user
            User.objects.create_user(username=username, email=email, password=password)
            return redirect('login')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'register.html', {'form': form})

def login_view(request):
    """
    Standard login view - uses Django authentication.
    """
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # clear admin_authenticated session flag on login (to require re-auth if switching users)
            try:
                request.session.pop('admin_authenticated', None)
            except Exception:
                pass
            return redirect('form')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')

def logout_view(request):
    """
    Logout and clear admin session flag.
    """
    try:
        request.session.pop('admin_authenticated', None)
    except Exception:
        pass
    logout(request)
    return redirect('login')

def input_form(request):
    """
    Profile input form - collects profile features and redirects to chat (prediction).
    """
    if request.method == 'POST':
        form = ProfileForm(request.POST)
        if form.is_valid():
            # Save to session to be processed in chat_view
            request.session['form_data'] = {
                'followers': form.cleaned_data['followers'],
                'following': form.cleaned_data['following'],
                'bio': form.cleaned_data['bio'],
                'has_profile_photo': int(form.cleaned_data['has_profile_photo']),
                'is_private': int(form.cleaned_data['is_private']),
            }
            return redirect('chat')
    else:
        form = ProfileForm()
    return render(request, 'form.html', {'form': form})

# ========== PREDICTION & DETECTION VIEWS ==========

def generate_pie_chart(predictions):
    labels = ['Fake', 'Real']
    counts = [predictions.count('Fake'), predictions.count('Real')]

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return pie_chart

def generate_bar_chart(predictions):
    labels = ['Fake', 'Real']
    counts = [predictions.count('Fake'), predictions.count('Real')]

    fig, ax = plt.subplots()
    ax.bar(labels, counts)
    ax.set_ylabel('Count')
    ax.set_title('Prediction Counts')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    bar_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return bar_chart

def chat_view(request):
    rf_prediction = svm_prediction = ann_prediction = consensus = None
    pie_chart = bar_chart = None
    detection_id = None
    error_message = None

    # Load saved models & scaler
    try:
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        with open('ann_model.pkl', 'rb') as f:
            ann_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        models_loaded = True
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        models_loaded = False
        error_message = "AI models are not available. Please contact administrator."
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False
        error_message = f"Error loading AI models: {str(e)}"

    if not models_loaded:
        return render(request, 'chat.html', {
            'rf_prediction': 'Error',
            'svm_prediction': 'Error',
            'ann_prediction': 'Error',
            'consensus': 'Model Error',
            'pie_chart': None,
            'bar_chart': None,
            'detection_id': None,
            'error_message': error_message
        })

    if request.method == 'POST':
        try:
            followers = int(request.POST.get('followers', 0))
            following = int(request.POST.get('following', 0))
            bio = request.POST.get('bio', '')
            bio_length = len(bio)
            profile_photo = int(request.POST.get('has_profile_photo', 0))
            is_private = int(request.POST.get('is_private', 0))

            # Validate input ranges
            if followers < 0 or following < 0:
                raise ValueError("Followers and following counts cannot be negative")
            if profile_photo not in [0, 1] or is_private not in [0, 1]:
                raise ValueError("Invalid binary field value")

            features = [followers, following, bio_length, profile_photo, is_private]
            features_scaled = scaler.transform([features])

            rf_pred = rf_model.predict(features_scaled)[0]
            svm_pred = svm_model.predict(features_scaled)[0]
            ann_pred = ann_model.predict(features_scaled)[0]

            rf_prediction = 'Fake' if rf_pred == 1 else 'Real'
            svm_prediction = 'Fake' if svm_pred == 1 else 'Real'
            ann_prediction = 'Fake' if ann_pred == 1 else 'Real'

            predictions = [rf_prediction, svm_prediction, ann_prediction]
            consensus = max(set(predictions), key=predictions.count)

            fake_count = predictions.count('Fake')
            confidence_score = fake_count / len(predictions)

            # Save detection
            try:
                detection = Detection(
                    profile_username=f"user_input_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    followers=followers,
                    following=following,
                    bio_length=bio_length,
                    has_profile_photo=bool(profile_photo),
                    is_private=bool(is_private),
                    rf_prediction=rf_prediction,
                    svm_prediction=svm_prediction,
                    ann_prediction=ann_prediction,
                    consensus=consensus,
                    confidence_score=confidence_score
                )
                detection.save()
                detection_id = detection.id
            except Exception as db_error:
                print(f"Database error: {db_error}")
                detection_id = None

            # Generate charts
            try:
                pie_chart = generate_pie_chart(predictions)
                bar_chart = generate_bar_chart(predictions)
            except Exception as chart_error:
                print(f"Chart generation error: {chart_error}")
                pie_chart = None
                bar_chart = None

        except ValueError as ve:
            print(f"Input validation error: {ve}")
            error_message = f"Invalid input: {str(ve)}"
            rf_prediction = svm_prediction = ann_prediction = "Error"
            consensus = "Input Error"
        except Exception as e:
            print(f"Error during prediction: {e}")
            error_message = f"Prediction error: {str(e)}"
            rf_prediction = svm_prediction = ann_prediction = "Error"
            consensus = "System Error"

    else:
        # GET: process form_data from session if present
        form_data = request.session.pop('form_data', None)
        if form_data:
            try:
                followers = form_data.get('followers', 0)
                following = form_data.get('following', 0)
                bio = form_data.get('bio', '')
                bio_length = len(bio)
                profile_photo = form_data.get('has_profile_photo', 0)
                is_private = form_data.get('is_private', 0)

                if followers < 0 or following < 0:
                    raise ValueError("Invalid input values")
                if profile_photo not in [0, 1] or is_private not in [0, 1]:
                    raise ValueError("Invalid binary field value")

                features = [followers, following, bio_length, profile_photo, is_private]
                features_scaled = scaler.transform([features])

                rf_pred = rf_model.predict(features_scaled)[0]
                svm_pred = svm_model.predict(features_scaled)[0]
                ann_pred = ann_model.predict(features_scaled)[0]

                rf_prediction = 'Fake' if rf_pred == 1 else 'Real'
                svm_prediction = 'Fake' if svm_pred == 1 else 'Real'
                ann_prediction = 'Fake' if ann_pred == 1 else 'Real'

                predictions = [rf_prediction, svm_prediction, ann_prediction]
                consensus = max(set(predictions), key=predictions.count)

                fake_count = predictions.count('Fake')
                confidence_score = fake_count / len(predictions)

                # Save detection
                try:
                    detection = Detection(
                        profile_username=f"user_input_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        followers=followers,
                        following=following,
                        bio_length=bio_length,
                        has_profile_photo=bool(profile_photo),
                        is_private=bool(is_private),
                        rf_prediction=rf_prediction,
                        svm_prediction=svm_prediction,
                        ann_prediction=ann_prediction,
                        consensus=consensus,
                        confidence_score=confidence_score
                    )
                    detection.save()
                    detection_id = detection.id
                except Exception as db_error:
                    print(f"Database error in GET: {db_error}")
                    detection_id = None

                # Charts
                try:
                    pie_chart = generate_pie_chart(predictions)
                    bar_chart = generate_bar_chart(predictions)
                except Exception as chart_error:
                    print(f"Chart generation error in GET: {chart_error}")
                    pie_chart = None
                    bar_chart = None
            except Exception as e:
                print(f"Error during GET prediction: {e}")
                error_message = f"Error processing form data: {str(e)}"
                rf_prediction = svm_prediction = ann_prediction = "Error"
                consensus = "Processing Error"

    return render(request, 'chat.html', {
        'rf_prediction': rf_prediction,
        'svm_prediction': svm_prediction,
        'ann_prediction': ann_prediction,
        'consensus': consensus,
        'pie_chart': pie_chart,
        'bar_chart': bar_chart,
        'detection_id': detection_id,
        'error_message': error_message
    })

# ========== REPORTING VIEWS ==========

@login_required
def submit_report(request):
    """Allow users to submit reports after detection"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    initial_data = {}
    error_message = None
    success_message = None
    
    # Get pre-filled data from URL parameters
    report_type = request.GET.get('type', '')
    detection_id = request.GET.get('detection_id', '')
    
    # Validate detection_id
    valid_detection_id = None
    if detection_id and detection_id != 'None':
        try:
            valid_detection_id = int(detection_id)
            # Verify the detection actually exists
            try:
                detection = Detection.objects.get(id=valid_detection_id)
                initial_data['detection_data'] = detection
            except Detection.DoesNotExist:
                error_message = "The referenced detection was not found."
                valid_detection_id = None
        except (ValueError, TypeError):
            error_message = "Invalid detection ID format."
            valid_detection_id = None
    
    if report_type:
        initial_data['report_type'] = report_type
    
    if request.method == 'POST':
        form = UserReportForm(request.POST)
        if form.is_valid():
            try:
                report = form.save(commit=False)
                report.user = request.user
                
                # Link detection if provided and valid
                if valid_detection_id:
                    try:
                        detection = Detection.objects.get(id=valid_detection_id)
                        report.detection_data = detection
                    except Detection.DoesNotExist:
                        print(f"Detection {valid_detection_id} not found when saving report")
                
                report.save()
                success_message = "Thank you! Your report has been submitted successfully."
                
                # Clear the form after successful submission
                form = UserReportForm()
                
                # If it was an AJAX request, return JSON response
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'success', 
                        'message': success_message
                    })
                
            except Exception as e:
                error_message = f"An error occurred while saving your report: {str(e)}"
                print(f"Error saving report: {e}")
        else:
            error_message = "Please correct the errors in the form."
    else:
        form = UserReportForm(initial=initial_data)
    
    context = {
        'form': form,
        'detection_id': valid_detection_id,
        'error_message': error_message,
        'success_message': success_message,
        'report_type': report_type,
    }
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        if error_message:
            return JsonResponse({'status': 'error', 'message': error_message})
        return JsonResponse({'status': 'success', 'message': 'Form loaded successfully'})
    
    return render(request, 'submit_report.html', context)

# ---------- UPDATED reports_list: requires admin session flag ----------
# Admin credential: username must be 'admin' (i.e., logged in), and admin password to enter reports view is hard-coded as 'admin123'.
# After successful POST the session key 'admin_authenticated' is set True.
@login_required(login_url='/login/')
def reports_list(request):
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin123'
    admin_authenticated = request.session.get('admin_authenticated', False)

    # Ensure an 'admin' user exists (development convenience)
    try:
        admin_user = User.objects.filter(username=ADMIN_USERNAME).first()
        if admin_user is None:
            admin_user = User.objects.create_user(username=ADMIN_USERNAME, password=ADMIN_PASSWORD)
            admin_user.is_staff = True
            admin_user.is_superuser = True
            admin_user.save()
    except Exception as e:
        # Do not crash - log error
        print(f"Could not auto-create admin user: {e}")

    # If the logged-in user is not admin, deny access
    if request.user.username != ADMIN_USERNAME:
        return render(request, 'permission_denied.html', status=403)

    # If admin not yet authenticated for reports, show the admin password form
    login_error = None
    if request.method == 'POST' and not admin_authenticated:
        provided = request.POST.get('admin_password', '')
        if provided == ADMIN_PASSWORD:
            request.session['admin_authenticated'] = True
            admin_authenticated = True
            return redirect('reports_list')
        else:
            login_error = "Invalid admin password."

    if request.user.username == ADMIN_USERNAME and not admin_authenticated:
        # Render template that includes csrf_token and password input (templates/admin_reports_login.html)
        return render(request, 'admin_reports_login.html', {'error': login_error})

    # Admin is authenticated; show all reports
    user_reports = UserReport.objects.all().order_by('-timestamp')

    # Statistics
    total_reports = user_reports.count()
    fake_reports = user_reports.filter(report_type='fake').count()
    real_reports = user_reports.filter(report_type='real').count()
    suspicious_reports = user_reports.filter(report_type='suspicious').count()
    resolved_reports = user_reports.filter(is_resolved=True).count()
    
    context = {
        'reports': user_reports,
        'total_reports': total_reports,
        'fake_reports': fake_reports,
        'real_reports': real_reports,
        'suspicious_reports': suspicious_reports,
        'resolved_reports': resolved_reports,
    }
    
    return render(request, 'reports_list.html', context)

# ========== ADMIN ANALYTICS VIEWS ==========

@staff_member_required
def admin_analytics(request):
    days = int(request.GET.get('days', 7))
    start_date = timezone.now() - timedelta(days=days)
    
    all_detections = Detection.objects.filter(timestamp__gte=start_date)
    total_detections = all_detections.count()
    fake_detections = all_detections.filter(consensus='Fake')
    real_detections = all_detections.filter(consensus='Real')
    
    fake_count = fake_detections.count()
    real_count = real_detections.count()
    
    pending_reviews = fake_detections.filter(admin_verification='Pending').count()
    confirmed_fake = fake_detections.filter(admin_verification='Confirmed').count()
    false_positives = fake_detections.filter(admin_verification='False Positive').count()
    
    rf_fake = all_detections.filter(rf_prediction='Fake').count()
    svm_fake = all_detections.filter(svm_prediction='Fake').count()
    ann_fake = all_detections.filter(ann_prediction='Fake').count()
    
    recent_detections = fake_detections.order_by('-timestamp')[:10]
    
    detection_chart = generate_detection_chart(fake_count, real_count)
    model_comparison_chart = generate_model_comparison_chart(rf_fake, svm_fake, ann_fake, total_detections)
    timeline_chart = generate_timeline_chart(days)
    
    context = {
        'total_detections': total_detections,
        'fake_count': fake_count,
        'real_count': real_count,
        'fake_percentage': (fake_count / total_detections * 100) if total_detections > 0 else 0,
        'pending_reviews': pending_reviews,
        'confirmed_fake': confirmed_fake,
        'false_positives': false_positives,
        'recent_detections': recent_detections,
        'days': days,
        'detection_chart': detection_chart,
        'model_comparison_chart': model_comparison_chart,
        'timeline_chart': timeline_chart,
    }
    
    return render(request, 'admin_analytics.html', context)

@staff_member_required
def admin_review_detections(request):
    if request.method == 'POST':
        detection_id = request.POST.get('detection_id')
        action = request.POST.get('action')
        
        detection = Detection.objects.get(id=detection_id)
        detection.reviewed_by_admin = True
        detection.admin_verification = action
        detection.save()
        
        return redirect('admin_review_detections')
    
    unreviewed_detections = Detection.objects.filter(
        consensus='Fake', 
        reviewed_by_admin=False
    ).order_by('-timestamp')
    
    all_detections = Detection.objects.filter(consensus='Fake').order_by('-timestamp')
    
    context = {
        'unreviewed_detections': unreviewed_detections,
        'all_detections': all_detections,
    }
    
    return render(request, 'admin_review.html', context)

# Chart generation functions for admin analytics
def generate_detection_chart(fake_count, real_count):
    labels = ['Fake Profiles', 'Real Profiles']
    sizes = [fake_count, real_count]
    colors = ['#ff6b6b', '#51cf66']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Profile Detection Distribution')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return chart

def generate_model_comparison_chart(rf_fake, svm_fake, ann_fake, total):
    labels = ['Random Forest', 'SVM', 'ANN']
    fake_counts = [rf_fake, svm_fake, ann_fake]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, fake_counts)
    ax.set_ylabel('Number of Fake Detections')
    ax.set_title('Model Performance Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height} ({height/total*100:.1f}%)' if total > 0 else '0',
                ha='center', va='bottom')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return chart

def generate_timeline_chart(days):
    dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
    fake_counts = [np.random.randint(5, 20) for _ in range(days)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([d.strftime('%m/%d') for d in dates], fake_counts, marker='o', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Fake Detections')
    ax.set_title(f'Fake Profile Detections Over {days} Days')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return chart
