from django import forms
from .models import ProfileData, UserReport

class ProfileForm(forms.ModelForm):
    class Meta:
        model = ProfileData
        fields = ['followers', 'following', 'bio', 'has_profile_photo', 'is_private']
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 3, 'cols': 40}),
        }

class UserRegistrationForm(forms.Form):
    username = forms.CharField(max_length=100)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput())

# Add this new form for user reports
class UserReportForm(forms.ModelForm):
    class Meta:
        model = UserReport
        fields = ['detected_profile', 'report_type', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Provide additional details about why you think this account is fake/real...'}),
            'detected_profile': forms.TextInput(attrs={'placeholder': 'Enter username or profile URL'}),
        }
        labels = {
            'detected_profile': 'Profile Username/URL',
            'report_type': 'Report Type',
            'description': 'Additional Details',
        }