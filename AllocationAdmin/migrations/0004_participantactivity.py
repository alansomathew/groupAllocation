# Generated by Django 5.0.3 on 2024-04-24 13:52

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AllocationAdmin', '0003_remove_participant_event'),
    ]

    operations = [
        migrations.CreateModel(
            name='ParticipantActivity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('activity', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='AllocationAdmin.event')),
                ('participant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='AllocationAdmin.participant')),
            ],
        ),
    ]
