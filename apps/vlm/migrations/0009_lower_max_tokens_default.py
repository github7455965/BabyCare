from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vlm', '0008_vlmqueuedtask'),
    ]

    operations = [
        migrations.AlterField(
            model_name='vlmpromptconfig',
            name='max_tokens',
            field=models.IntegerField(
                default=512,
                help_text='VLM 单次回答最大 token 数（默认 512 覆盖状态描述；判断型可调小，描述型可调大）',
                verbose_name='VLM max_tokens',
            ),
        ),
    ]
