# myapp/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    occupation = models.CharField(max_length=100, blank=True)
    usage = models.CharField(max_length=100, blank=True)
    bio = models.TextField(blank=True)

    def __str__(self):
        return self.user.username



class EmailOTP(models.Model):
    """
    OTP storage: can be tied to a user (existing) or kept as an email string for signup flows.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)   # allow storing OTP for an email before user exists
    code = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    is_used = models.BooleanField(default=False)

    def is_expired(self):
        """10 minute expiry"""
        return (timezone.now() - self.created_at).total_seconds() > 600

    def __str__(self):
        return f"{self.email or (self.user.email if self.user else 'unknown')} - {self.code}"

class Notes(models.Model):
    """
    Unified Notes model used for saved summaries (text / audio / video / qna / quiz).
    content: store JSON string (summary payload) or raw text.
    """
    NOTE_TYPES = [
        ('text', 'Text'),
        ('audio', 'Audio'),
        ('video', 'Video'),
        ('qna', 'Q&A'),
        ('quiz', 'Quiz'),
        ('other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="notes")
    title = models.CharField(max_length=200, blank=True)
    content = models.TextField()            # store either raw text OR JSON stringified payload
    note_type = models.CharField(max_length=20, choices=NOTE_TYPES, default='text')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title or 'Note'} ({self.user.username})"

    @property
    def preview(self):
        return (self.content[:200] + '...') if len(self.content) > 200 else self.content

    def get_summary(self):
        """If content is a JSON string (summary payload), return parsed dict; else return {'text': content}"""
        try:
            return json.loads(self.content)
        except Exception:
            return {"text": self.content}
