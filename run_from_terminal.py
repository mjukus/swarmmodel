# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:03:26 2021

@author: thesq
This code handles the command line running of the program, accounting for required and optional parameters. Needlessly complicated? Maybe.
"""

from main import main # the real program
import sys # imports system functions for handling terminal input
from dataclasses import dataclass # new class definition module, only works in python 3.7 up. REQUIRED.

help_text = "For help and more information on optional arguments, pass final argument '?'.\n" # saves a bit of repetition in the error strings
typesDict = {"rodLength" : float, "partMass" : float, "lennardJonesFlag" : bool, "epsilon" : float, "sigma" : float, "forceCap" : float, "hydrodynamics" : bool, "swimmingSpeed" : float, "tumbleFreq" : int, "hydrodynamicThrust" : float, "viscosity" : float} # dictionary of keyword arguments of main() and their types

@dataclass
class integer:
    '''These classes are all pretty similar so are only commented where something new is occurring. They describe string objects which are to be converted to other data types.'''
    name: str # named to make the exceptions as easy to understand as possible
    value: str # counterintuitive, but the input is a string, so value is too (for now)
    
    def inputCheck(self,minimum: int=1):
        '''
        Tries to convert the instance to an integer and checks that it is larger than a specified minimum.

        Parameters
        ----------
        minimum : int, optional
            The minimum allowed value of an input integer. The default is 1.
        '''
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
        '''
        Tries to convert the instance to a float and checks that it is positive.
        '''
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
        '''
        Tries to convert the instance to a boolean.
        '''
        if self.value == "True":
            self.value = True
        elif self.value == "False":
            self.value = False
        else:
            raise TypeError(f"{self.name} must be a boolean (True/False).")

def kwargsProduction(inputString: str):
    '''
    Takes an input string of a user-specified argument and uses the parameter dictionary typesDict to convert the value to the correct type and add it to a list of keyword arguments to pass to a function.

    Parameters
    ----------
    inputString : str
        A string of the format "parameter=value".
    '''
    if not "=" in inputString:
        raise TypeError("Optional arguments must be specified in the form <optionalArgument=value>. Spaces should only be placed between arguments.\n")
    stringParts = inputString.split("=") # split string into parts separated by "="
    if typesDict[stringParts[0]] == float:
        # if/elif/else statement that searches the dictionary typesDict for the parameter before the "=" and creates an instance of the appropriate class for that parameter.
        quantity = positiveFloat(stringParts[0],stringParts[1])
    elif typesDict[stringParts[0]] == bool:
        quantity = boolean(stringParts[0],stringParts[1])
    elif typesDict[stringParts[0]] == int:
        # currently unnecessary - no integer kwargs - but included in case that changes
        quantity = integer(stringParts[0],stringParts[1])
    else:
        raise TypeError("{inputString} is not a valid argument.\n{help_text}")
    quantity.inputCheck() # checks the value is the right input type and value and converts it
    kwargs[stringParts[0]] = quantity.value # appends parameter and value to the dictionary of keyword arguments

if sys.argv[-1] == '?':
    #provides help if the user adds "?" to the end of their input. Bit messy
    help(main)
    print("\nAdd optional arguments using the syntax <optionalArgument=value> after required arguments.")
    args = sys.argv[0:-1] # new list without "?"
else:
    args = sys.argv
if len(args) - 1 < 5:
    # checks number of arguments passed to see if the input is of a valid length
    raise TypeError(f"main() takes a minimum of 5 arguments ({len(args) - 1} given).\n\nUsage: python {args[0]} <axisN> <nRod> <partAxisSep> <Nt> <timestep>.\n{help_text}")
elif len(args) - 1 > 15:
    raise TypeError(f"main() takes a maximum of 15 arguments ({len(args) - 1} given).\n{help_text}")

axisN = integer("axisN",args[1]) # these lines make instances of the relevant classes for each required argument
nRod = integer("nRod",args[2])
partAxisSep = positiveFloat("partAxisSep",args[3])
Nt = integer("Nt",args[4])
timestep = positiveFloat("timestep",args[5])

axisN.inputCheck() # and check the inputs are of the right form
nRod.inputCheck(2)
partAxisSep.inputCheck()
Nt.inputCheck(0)
timestep.inputCheck()

kwargs = {}
for i in args[6:]:
    # iterates over remaining arguments to produce a dictionary of keyword arguments
    kwargsProduction(i)

main(axisN.value,nRod.value,partAxisSep.value,Nt.value,timestep.value,**kwargs) # calls the real program with all the parameters the user specified