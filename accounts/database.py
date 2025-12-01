import os
import django
import csv

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_profile_detect.settings')
django.setup()

from accounts.models import ProfileData
from django.contrib.auth.models import User
from django.db import connection

def view_all_profiles():
    print("\n--- All Profiles ---")
    all_profiles = ProfileData.objects.all()
    for profile in all_profiles:
        print(f"Username: {profile.user.username}, Followers: {profile.followers}, Following: {profile.following}, Bio: {profile.bio}")

def filter_profiles():
    print("\n--- Filtered Profiles (Followers > 100) ---")
    popular_profiles = ProfileData.objects.filter(followers__gt=100)
    for profile in popular_profiles:
        print(f"Username: {profile.user.username}, Followers: {profile.followers}")

def add_new_profile():
    print("\n--- Adding a New Profile ---")
    user = User.objects.create(username="john_doe", email="john@example.com")
    profile = ProfileData.objects.create(
        user=user,
        followers=120,
        following=150,
        bio="Hello, this is John's profile.",
        has_profile_photo=True,
        is_private=False
    )
    print(f"Profile created: {profile.user.username}")

def update_profile():
    print("\n--- Updating a Profile ---")
    profile = ProfileData.objects.get(user__username="john_doe")
    profile.bio = "Updated bio for John."
    profile.followers = 200
    profile.save()
    print(f"Profile updated: {profile.bio}, Followers: {profile.followers}")

def delete_profile():
    print("\n--- Deleting a Profile ---")
    profile = ProfileData.objects.get(user__username="john_doe")
    profile.delete()
    print("Profile deleted.")

def raw_sql_query():
    print("\n--- Raw SQL Query ---")
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM accounts_profiledata;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

def export_to_csv():
    print("\n--- Exporting Profiles to CSV ---")
    with open('profiledata.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Username', 'Followers', 'Following', 'Bio', 'Has Profile Photo', 'Is Private'])
        for profile in ProfileData.objects.all():
            writer.writerow([
                profile.user.username,
                profile.followers,
                profile.following,
                profile.bio,
                profile.has_profile_photo,
                profile.is_private
            ])
    print("Profiles exported to profiledata.csv")


def show_table_info():
    print("\n--- Table Info (PRAGMA) ---")
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(accounts_profiledata);")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

def main():
    print("Starting Django Database Operations...\n")
    view_all_profiles()
    filter_profiles()
    add_new_profile()
    update_profile()
    delete_profile()
    raw_sql_query()
    export_to_csv()
    show_table_info()

if __name__ == "__main__":
    main()