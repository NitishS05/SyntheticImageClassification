from django.db import models

from django.conf.urls.static import static
from django.conf import settings

# Create your models here.

def get_file_path(instance, filename):
    ext = filename.split('.')[-1]
    extension=ext
    filename = "%s.%s" % (instance.id, ext)
    return '{0}/{1}'.format('uploads/', filename)


class Image(models.Model):
	img= models.FileField(upload_to=get_file_path, null=True, blank= True )

	def __str__(self):
		return str(self.id)