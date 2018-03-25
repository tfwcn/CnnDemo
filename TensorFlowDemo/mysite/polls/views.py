from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import sys
sys.path.append("..")
from TestDemo import *

# Create your views here.
@csrf_exempt
def index(request):
    print("aaa")
    print(request.FILES)
    print("bbb")
    f1 = request.FILES["f1"]
    #file_path = os.path.join('', f1.name)
    #f = open(file_path, 'wb')
    #for chunk in f1.chunks():
    #    f.write(chunk)
    #f.close()
    run(f1.read())
    return HttpResponse('{"hello":"python"}', content_type="application/json")