from django import forms

class SignupForm(forms.Form):
    name = forms.CharField(label='Enter your name', max_length=100)
    email = forms.EmailField(label='Enter your email', max_length=100)

class selectChartForm(forms.Form):
    palabras = 'Frecuencia de palabras'
    idiomas = 'Frecuencia idiomas'
    chart_choices = (
        (palabras, "Frecuencia de palabras"),
        (idiomas, "Frecuencia idiomas")
    )
    chart = forms.ChoiceField(choices=chart_choices)