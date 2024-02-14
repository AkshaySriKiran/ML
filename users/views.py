# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'train.csv'
    import pandas as pd
    df = pd.read_csv(path, nrows=101)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


def data_preprocessed(request):
    from .utility.clickbait_preprocessed import pre_processed_data
    data = pre_processed_data()
    return render(request, 'users/preprocessed.html', {'data': data.to_html})


def user_ml_code(request):
    from .utility.clickbait_preprocessed import start_ml_procedeing
    lg_cr, rf_cr, gr_cr, en_cr = start_ml_procedeing()
    return render(request, 'users/ml_results.html', {'lg': lg_cr, 'rf': rf_cr, 'gr': gr_cr, 'en':en_cr})
