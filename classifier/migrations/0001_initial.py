# Generated by Django 4.0.1 on 2022-04-09 10:59

import classifier.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img', models.FileField(blank=True, null=True, upload_to=classifier.models.get_file_path)),
            ],
        ),
    ]
