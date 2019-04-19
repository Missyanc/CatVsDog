import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from catVdog.models import IMG
from Include.cnn.predict import predict

# Create your views here.
# 添加 index 函数，返回 index.html 页面
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def uploadImg(request):
    for file in os.listdir("E:/PycharmProjects/CatVsDog/media/img/"):
        targetFile = os.path.join("E:/PycharmProjects/CatVsDog/media/img/", file)
        if os.path.isfile(targetFile):
            os.remove(targetFile)
    if request.method == 'POST':
        new_img = IMG(
            img=request.FILES.get('img'),
            name=request.FILES.get('img').name
        )
        new_img.save()
    return render(request, 'uploadimg.html')

def result (request):
    result = predict()
    return render(request, 'result.html', {"data": result})
