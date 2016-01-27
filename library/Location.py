__author__ = 'Zixiao Shi'

class Location:
    def __init__(self, name, lon, lat, elev=0, timezone=None, lsm = None):
        self.name = name
        self.longitude = lon
        self.latitude = lat
        self.elevation = elev
        self.updateTimezone(timezone, lsm)


    def updateTimezone(self,timezone=None,lsm=None):
        if timezone == None:
            if self.longitude > 0:
                self.timezone = (self.longitude-7.5)//15
            else:
                self.timezone = (self.longitude+7.5)//15
        elif timezone <-12 or timezone > 12:
            raise ValueError("invalid time zone")
        else:
            self.timezone = timezone
        self.updateLSM(lsm)

    def updateLSM(self,lsm=None):
        if lsm == None:
            self.lsm = self.timezone * 15
        else:
            self.lsm = lsm