# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import regex
import os

from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError
from train import Trainer
from generate import Generator
from io_handler import IOHandler


style = style_from_dict({ #taken from the codeburst example and tweaked
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})


class PictureValidator(Validator): #checking if the folder has at least 1000 pics in it
    def validate(self, document):
        valid =  os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        exists = os.path.exists(document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        is_folder = os.path.isdir(document.text) 
        if not is_folder:
            raise ValidationError(
            message='The file path does not lead to a folder! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        has_1000 = False
        contents = len(os.listdir(document.text))
        if contents < 10:
            raise ValidationError(
            message='There is not enough files to properly train in this folder! Please try again:',
            cursor_position=len(document.text))  # Move cursor to end'


class PathValidator(Validator): #checking if is folder
    def validate(self, document):
        valid =  os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        exists = os.path.exists(document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        is_folder = os.path.isdir(document.text) 
        if not is_folder:
            raise ValidationError(
            message='The file path does not lead to a folder! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        



class FileValidator(Validator): #checking if is file and an image
    def validate(self, document):
        valid =  os.path.isabs(document.text)
        if not valid:
            raise ValidationError(
                message='Please enter the directory from the root:',
                cursor_position=len(document.text))  # Move cursor to end'
        exists = os.path.exists(document.text) 
        if not exists:
            raise ValidationError(
            message='The file path does not exist! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        is_file = os.path.isfile(document.text) 
        doc = os.path.basename(document.text)
        f_type = doc.split('.')
        is_img = True if f_type[1] == 'pdf' or f_type[1] == 'jpg' or f_type[1] == 'png' else False
        if not (is_file and is_img):
            raise ValidationError(
            message='The file path does not lead to an image! Please enter a valid path:',
            cursor_position=len(document.text))  # Move cursor to end'
        


class App():
    def __init__(self):
        print('Welcome to ImpressiGAN!')
        self.io = IOHandler()
        image_shape = (256, 256, 3)
        self.gen = Generator()
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
                    },
                    {
                        'name': 'See Cool Pics'
                    },
                ],
                'validate': lambda answer: 'You must choose an option' \
                    if len(answer) == 0 else True
            }
        ]
        answer = prompt(questions, style = style)
        print(answer['actions'])
        if answer['actions'] == ['Train a Model (this will take a while!)']:
            self.train()
        elif answer['actions'] == ['Generate a Picture']:
            self.generate()
        else:
            print("butthole")


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
        answers = prompt(questions)
        print('Converting now...')
        self.gen.generate(answers['weights'], answers['src image'], answers['target'])
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
        answers = prompt(questions)
        print('Training now...')
        self.trainer.train(answers['src images'], answers['src art'])

app = App()


# print('Welcome to ImpressiGAN!')

# questions = [
#     {
#         'type': 'checkbox',
#         'message': 'What do you want to do?',
#         'name': 'actions',
#         'choices': [
#             Separator('= Actions ='),
#             {
#                 'name': 'Train a Model (this will take a while!)'
#             },
#             {
#                 'name': 'Generate a Picture'
#             },
#             {
#                 'name': 'See Cool Pics'
#             },
#         ],
#         'validate': lambda answer: 'You must choose an option' \
#             if len(answer) == 0 else True
#     },
#     {
#         'type': 'input',
#         'name': 'src image',
#         'message': 'Please enter the file path to the image you want to convert from the root directory!',
#         'validate': FileValidator
#         'when': lambda answers: answers['actions'] == 'Train a Model (this will take a while!)'
#     },
#     {
#         'type': 'list',
#         'name': 'size',
#         'message': 'What size do you need?',
#         'choices': ['Large', 'Medium', 'Small'],
#         'filter': lambda val: val.lower()
#     },
#     {
#         'type': 'input',
#         'name': 'quantity',
#         'message': 'How many do you need?',
#         'validate': NumberValidator,
#         'filter': lambda val: int(val)
#     },
#     {
#         'type': 'expand',
#         'name': 'toppings',
#         'message': 'What about the toppings?',
#         'choices': [
#             {
#                 'key': 'p',
#                 'name': 'Pepperoni and cheese',
#                 'value': 'PepperoniCheese'
#             },
#             {
#                 'key': 'a',
#                 'name': 'All dressed',
#                 'value': 'alldressed'
#             },
#             {
#                 'key': 'w',
#                 'name': 'Hawaiian',
#                 'value': 'hawaiian'
#             }
#         ]
#     },
#     {
#         'type': 'rawlist',
#         'name': 'beverage',
#         'message': 'You also get a free 2L beverage',
#         'choices': ['Pepsi', '7up', 'Coke']
#     },
#     {
#         'type': 'input',
#         'name': 'comments',
#         'message': 'Any comments on your purchase experience?',
#         'default': 'Nope, all good!'
#     },
#     {
#         'type': 'list',
#         'name': 'prize',
#         'message': 'For leaving a comment, you get a freebie',
#         'choices': ['cake', 'fries'],
#         'when': lambda answers: answers['comments'] != 'Nope, all good!'
#     }
# ]

# answers = prompt(questions, style=style)
# print('Order receipt:')
# pprint(answers)