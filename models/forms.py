# forms.py
from django import forms

class StudentDataForm(forms.Form):
    gender_choices = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]

    ethnicity_choices = [
        ('group A', 'Group A'),
        ('group B', 'Group B'),
        ('group C', 'Group C'),
        ('group D', 'Group D'),
        ('group E', 'Group E')
    ]

    education_choices = [
        ("bachelor's degree", "Bachelor's Degree"),
        ("some college", "Some College"),
        ("associate's degree","Associate's degree"),
        ("high school", "High School"),
        ("master's degree", "Master's degree"),
        ("some high school", "Some high school"),
      
    ]

    lunch_choices = [
        ('standard', 'Standard'),
        ('free/reduced', 'Free/Reduced'),
    ]

    test_prep_choices = [
        ('none', 'None'),
        ('completed', 'Completed'),
    ]

    gender = forms.ChoiceField(choices=gender_choices)
    race_ethnicity = forms.ChoiceField(choices=ethnicity_choices, label='Race/Ethnicity')
    parental_education = forms.ChoiceField(choices=education_choices, label='Parental Level of Education')
    lunch = forms.ChoiceField(choices=lunch_choices)
    test_prep_course = forms.ChoiceField(choices=test_prep_choices, label='Test Preparation Course')
    reading_score = forms.IntegerField(label='Reading Score')
    writing_score = forms.IntegerField(label='Writing Score')
