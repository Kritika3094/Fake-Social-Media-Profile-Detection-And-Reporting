from django.db import models
from django.contrib.auth.models import User

class ProfileData(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    followers = models.IntegerField(default=0)
    following = models.IntegerField(default=0)
    bio = models.TextField(blank=True)
    has_profile_photo = models.BooleanField(default=False)
    is_private = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

class Report(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reported_profile = models.CharField(max_length=255)
    reason = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    blockchain_tx_hash = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"Report by {self.user.username} on {self.reported_profile}"

class Detection(models.Model):
    VERIFICATION_CHOICES = [
        ('Pending', 'Pending'),
        ('Confirmed', 'Confirmed'),
        ('False Positive', 'False Positive'),
    ]
    
    profile_username = models.CharField(max_length=255)
    followers = models.IntegerField()
    following = models.IntegerField()
    bio_length = models.IntegerField()
    has_profile_photo = models.BooleanField()
    is_private = models.BooleanField()
    rf_prediction = models.CharField(max_length=10)
    svm_prediction = models.CharField(max_length=10)
    ann_prediction = models.CharField(max_length=10)
    consensus = models.CharField(max_length=10)
    confidence_score = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    reviewed_by_admin = models.BooleanField(default=False)
    admin_verification = models.CharField(
        max_length=15, 
        choices=VERIFICATION_CHOICES,
        default='Pending'
    )

    def __str__(self):
        return f"Detection: {self.profile_username} - {self.consensus}"

# Add this new model for user reports after detection
class UserReport(models.Model):
    REPORT_TYPES = [
        ('fake', 'Fake Account'),
        ('real', 'Real Account'),
        ('suspicious', 'Suspicious Activity'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    detected_profile = models.CharField(max_length=255)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    description = models.TextField(blank=True)
    detection_data = models.ForeignKey(Detection, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)
    admin_notes = models.TextField(blank=True)

    def __str__(self):
        return f"Report by {self.user.username} - {self.detected_profile} ({self.report_type})"