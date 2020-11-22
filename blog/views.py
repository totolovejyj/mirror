from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import open
def one(request):
	if request.is_ajax():
		f = open("demofile.txt", "r") #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt", "r")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})

	else:
		return render(request, 'blog/one.html')

def open(request):
	open.function()
	return 




def home(request):
    
    return render(request, 'blog/home.html')


def about(request):
    return render(request, 'blog/about.html', {'title': 'Hello'})
    #title 은 valuable name , about 은 value
    #about 을 부르면 '.html'을 나타냄 , 그리고 about은 urls에서 부름


def abc(request):
	return render(request, 'blog/home.html')