from django import forms

CHOICES = [(0, 'Never'), (1, 'Almost Never'), (2, 'Sometimes'), (3, 'Fairly Often'), (4, 'Very Often')]

class BrainScoreForm(forms.Form):
    q1 = forms.ChoiceField(choices=CHOICES, label="Frequent morning headaches?", widget=forms.Select())
    q2 = forms.ChoiceField(choices=CHOICES, label="Changes in vision?", widget=forms.Select())
    q3 = forms.ChoiceField(choices=CHOICES, label="Experienced seizures?", widget=forms.Select())
    q4 = forms.ChoiceField(choices=CHOICES, label="Personality or behavior changes?", widget=forms.Select())
    q5 = forms.ChoiceField(choices=CHOICES, label="Weakness/numbness in arms or legs?", widget=forms.Select())
    q6 = forms.ChoiceField(choices=CHOICES, label="Difficulty speaking/understanding?", widget=forms.Select())
    q7 = forms.ChoiceField(choices=CHOICES, label="Changes in coordination or balance?", widget=forms.Select())
    q8 = forms.ChoiceField(choices=CHOICES, label="Nausea/vomiting worsening?", widget=forms.Select())
    q9 = forms.ChoiceField(choices=CHOICES, label="Unexplained weight loss/appetite loss?", widget=forms.Select())
    q10 = forms.ChoiceField(choices=CHOICES, label="Other unusual symptoms?", widget=forms.Select())
