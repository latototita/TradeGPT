from django.contrib import admin
from django.urls import path  , include, re_path
from django.conf.urls.static import static
from . import settings
from django.conf.urls import url
from django.views.static import serve
from django.conf.urls import (
                                handler400,
                                handler403,
                                handler404,
                                handler500)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('tasks.urls')),
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
]
#if settings.DEBUG:
urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL,
                              document_root=settings.STATIC_ROOT)

'''
handler400 = 'tasks.views.bad_request'
handler403 = 'tasks.views.permission_denied'
handler404 = 'tasks.views.page_not_found'
handler500 = 'tasks.views.server_error'
'''