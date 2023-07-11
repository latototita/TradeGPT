from django.db import  models
'''
class Predictions(models.Model):
    time = models.CharField(max_length=100)
    pips =models.CharField(max_length=100,default=1,blank=True)
    duration = models.CharField(max_length=100)
    trend =models.CharField(max_length=100,default=1,blank=True)

    def __str__(self):
        return self.trend
class Pairs(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Timeframes(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
'''