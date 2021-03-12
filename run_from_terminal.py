# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:03:26 2021

@author: thesq
"""

from main import main
import sys
from dataclasses import dataclass

help_text = "For help and more information on optional arguments, pass final argument '?'.\n"
typesDict = {"rodLength" : float, "partMass" : float, "lennardJonesFlag" : bool, "epsilon" : float, "sigma" : float, "forceCap" : float, "hydrodynamics" : bool, "swimmingSpeed" : float, "hydrodynamicThrust" : float, "viscosity" : float}

@dataclass
class integer:
    name: str
    value: str
    
    def inputCheck(self,minimum=1):
        try:
            self.value = int(self.value)
        except:
            raise TypeError(f"{self.name} must be an integer.")
        if self.value < minimum:
            raise ValueError(f"{self.name} only takes integers {minimum} or larger ({self.value} given).")

@dataclass
class positiveFloat:
    name: str
    value: str
    
    def inputCheck(self):
        try:
            self.value = float(self.value)
        except:
            raise TypeError(f"{self.name} must be a float.")
        if self.value <= 0:
            raise ValueError(f"{self.name} only takes values greater than 0 ({self.value} given).")
            
@dataclass
class boolean:
    name: str
    value: str
    
    def inputCheck(self):
        if self.value == "True":
            self.value = True
        elif self.value == "False":
            self.value = False
        else:
            raise TypeError(f"{self.name} must be a boolean (True/False).")

def kwargsProduction(inputString):
    stringParts = inputString.split("=")
    if typesDict[stringParts[0]] == float:
        quantity = positiveFloat(stringParts[0],stringParts[1])
    elif typesDict[stringParts[0]] == bool:
        quantity = boolean(stringParts[0],stringParts[1])
    elif typesDict[stringParts[0]] == int:
        quantity = integer(stringParts[0],stringParts[1])
    else:
        raise TypeError("{inputString} is not a valid argument.\n{help_text}")
    quantity.inputCheck()
    kwargs[stringParts[0]] = quantity.value

if sys.argv[-1] == '?':
    help(main)
    print("\nAdd optional arguments using the syntax <optionalArgument=value> after required arguments.")
    args = sys.argv[0:-1]
else:
    args = sys.argv
if len(args) < 5:
    raise TypeError(f"main() takes a minimum of 5 arguments ({len(args) - 1} given).\n\nUsage: python {args[0]} <axisN> <nRod> <partAxisSep> <Nt> <timestep>.\n{help_text}")
elif len(args) > 15:
    raise TypeError(f"main() takes a maximum of 15 arguments ({len(args) - 1} given).\n{help_text}")

axisN = integer("axisN",args[1])
nRod = integer("nRod",args[2])
partAxisSep = positiveFloat("partAxisSep",args[3])
Nt = integer("Nt",args[4])
timestep = positiveFloat("timestep",args[5])

axisN.inputCheck()
nRod.inputCheck(2)
partAxisSep.inputCheck()
Nt.inputCheck(0)
timestep.inputCheck()

kwargs = {}
for i in args[6:]:
    kwargsProduction(i)

main(axisN.value,nRod.value,partAxisSep.value,Nt.value,timestep.value,**kwargs)