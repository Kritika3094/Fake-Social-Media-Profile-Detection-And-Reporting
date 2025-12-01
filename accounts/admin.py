from django.contrib import admin
from .models import ProfileData, Report, Detection, UserReport

class ProfileDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'followers', 'following', 'has_profile_photo', 'is_private')
    search_fields = ('user__username',)

class ReportAdmin(admin.ModelAdmin):
    list_display = ('user', 'reported_profile', 'reason', 'timestamp', 'blockchain_tx_hash')
    search_fields = ('user__username', 'reported_profile')
    list_filter = ('timestamp',)

class DetectionAdmin(admin.ModelAdmin):
    list_display = ('profile_username', 'consensus', 'confidence_score', 'timestamp', 'reviewed_by_admin', 'admin_verification')
    list_filter = ('consensus', 'reviewed_by_admin', 'admin_verification', 'timestamp')
    search_fields = ('profile_username',)
    actions = ['mark_as_confirmed', 'mark_as_false_positive']
    
    def mark_as_confirmed(self, request, queryset):
        queryset.update(reviewed_by_admin=True, admin_verification='Confirmed')
    mark_as_confirmed.short_description = "Mark selected as Confirmed Fake"
    
    def mark_as_false_positive(self, request, queryset):
        queryset.update(reviewed_by_admin=True, admin_verification='False Positive')
    mark_as_false_positive.short_description = "Mark selected as False Positive"

class UserReportAdmin(admin.ModelAdmin):
    list_display = ('user', 'detected_profile', 'report_type', 'timestamp', 'is_resolved')
    list_filter = ('report_type', 'is_resolved', 'timestamp')
    search_fields = ('detected_profile', 'user__username')
    actions = ['mark_as_resolved', 'mark_as_unresolved']
    
    def mark_as_resolved(self, request, queryset):
        queryset.update(is_resolved=True)
    mark_as_resolved.short_description = "Mark selected as resolved"
    
    def mark_as_unresolved(self, request, queryset):
        queryset.update(is_resolved=False)
    mark_as_unresolved.short_description = "Mark selected as unresolved"

admin.site.register(ProfileData, ProfileDataAdmin)
admin.site.register(Report, ReportAdmin)
admin.site.register(Detection, DetectionAdmin)
admin.site.register(UserReport, UserReportAdmin)