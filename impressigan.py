# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import regex
import os

#from pyfiglet import Figlet
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError
from train import Trainer
from generate import Generator
from io_handler import IOHandler


style = style_from_dict({ #taken from the codeburst example and tweaked
    Token.QuestionMark: '#0970b0 bold',
    Token.Selected: '#9369ed bold', #slightly darker than question
    Token.Instruction: '#d3befa', 
    Token.Answer: '#6516f2 bold',
    Token.Question: '#9361ed',
})

WORKDIR = ''

class PictureValidator(Validator): #checking if the folder has at least 1000 pics in it
    def validate(self, document):
        valid = True# os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        global WORKDIR
        wdir = WORKDIR + '/' if WORKDIR else WORKDIR
        exists = os.path.exists(wdir + document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'

        is_folder = os.path.isdir(wdir + document.text) #checking if it is a directory

        if not is_folder:
            raise ValidationError(
            message='The file path does not lead to a folder! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'

        contents = len(os.listdir(wdir+document.text)) #and making sure it has at last 1000 photos in it
        if contents < 1000:
            raise ValidationError(
            message='There is not enough files to properly train in this folder! Please try again:',
            cursor_position=len(document.text))  # Move cursor to end'


class PathValidator(Validator): #checking if is folder
    def validate(self, document):
        valid = True# os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        global WORKDIR
        wdir = WORKDIR + '/' if WORKDIR else WORKDIR
        exists = os.path.exists(wdir + document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'

        is_folder = os.path.isdir(wdir+document.text) #but this time we just need to make sure it is a folder. Probs could've been combined
        if not is_folder:
            raise ValidationError(
            message='The file path does not lead to a folder! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        



class FileValidator(Validator): #checking if is file and an image
    def validate(self, document):
        valid = True #os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        global WORKDIR
        wdir = WORKDIR + '/' if WORKDIR else WORKDIR
        exists = os.path.exists(wdir + document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        
        files = os.listdir(wdir + document.text)
        for file in files:
            file = wdir + document.text + '/' + file
            is_file = os.path.isfile(file)  # check file
            doc = os.path.basename(file) #check file type
            f_type = doc.split('.')
            is_img = True if f_type[1] == 'pdf' or f_type[1] == 'jpg' or f_type[1] == 'png' else False
            if not (is_file and is_img):
                raise ValidationError(
                message='The file path does not lead to an image! Please enter a valid path:',
                cursor_position=len(document.text))  # Move cursor to end'

class App():
    def __init__(self):
        #this would have been fun 
      #  f = Figlet(font='speed')
      #  print(f.renderText('ImpressiGAN'))
        print('Welcome to ImpressiGAN!')
        answer = prompt([{
            'type': 'input',
            'name': 'workdir',
            'message': 'Please enter the your working directory!',
            'validate': PathValidator
        }])
        global WORKDIR
        WORKDIR = answer['workdir']
        print('now', WORKDIR)
        self.io = IOHandler(WORKDIR)
        image_shape = (256, 256, 3)
        self.gen = Generator(self.io)
        self.trainer = Trainer(image_shape, self.io)
        self.choose_action()
    
    def choose_action(self):
        questions = [
            {
                'type': 'checkbox',
                'message': 'What do you want to do?',
                'name': 'actions',
                'choices': [
                    Separator('= Actions ='),
                    {
                        'name': 'Train a Model (this will take a while!)'
                    },
                    {
                        'name': 'Generate a Picture'
                    }
                ],
                'validate': lambda answer: 'You must choose an option' \
                    if len(answer) == 0 else True
            }
        ]
        answer = prompt(questions, style = style) #this is what pulls the user answers and gives them to us in a dictionary
        print(answer['actions']) #so we take them
        if answer['actions'] == ['Train a Model (this will take a while!)']: #and check what they are
            self.train()
        elif answer['actions'] == ['Generate a Picture']:
            self.generate()
        else:
            print("Oops! Not a valid action.")


    def generate(self):
        questions = [ #this should be blended w Jake's thing
            {
                'type': 'list',
                'name': 'weights',
                'message': 'Which weight model would you like to use?',
                'choices': ['1', '2', '3'],
                #'validate': lambda val: val.lower()
            },

            {
                'type': 'input',
                'name': 'src image',
                'message': 'Please enter the file path to the image you want to convert from the root directory!',
                'validate': FileValidator
            },
            {
                'type': 'input',
                'name': 'target',
                'message': 'Please enter the file path to the folder you want your image in!',
                'validate': PathValidator
            }
        ]
        answers = prompt(questions, style = style) 
        print('Converting now...')
        self.gen.generate(answers['weights'], answers['src image'], answers['target']) #link to generarot facade
        print('Check your folder!')
    
    def train(self):
        questions = [ 
            {
                'type': 'input',
                'name': 'src images',
                'message': 'Please enter the file path to the images you want to train from!',
                'validate': PictureValidator
            },
                        {
                'type': 'input',
                'name': 'src art',
                'message': 'Please enter the file path to the art you want to train from!',
                'validate': PictureValidator
            }
        ]
        answers = prompt(questions, style = style)
        print('Training now...') #link to training facade
        self.trainer.train(answers['src images'], answers['src art'])

app = App()