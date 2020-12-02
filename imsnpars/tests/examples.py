'''
Created on 17.07.2018

@author: falensaa
'''

from tools import datatypes

oneWordExample = ("Collins",
                  datatypes.Tree([ -1 ]) )

msCollinsExample = (" Or was it because Ms. Collins had gone   ?",
                    datatypes.Tree([ 1, -1, 1, 1, 5, 6, 3, 6, 1 ]))

franceExample = ("    In France ? ? ! !",
                 datatypes.Tree([-1, 0, 0, 0, 0, 0]))

chamberMusicExample = ("Is this the future of chamber music ?",
                       datatypes.Tree([-1, 0, 3, 0, 3, 6, 4, 0]))

changesExample = ("Not all those who wrote oppose the changes .",
                  datatypes.Tree([1, 5, 1, 4, 1, -1, 7, 5, 5]))

# non-projective
hearingExample = ("A hearing is scheduled on the issue today .",
                  datatypes.Tree([1, 2, -1, 2, 1, 6, 4, 3, 2]))
    
afterExample =   ("That 's they 're what after",
                  datatypes.Tree([1, -1, 5, 4, 1, 4]))

letterExample = ("Who did you send the letter to ?",
                  datatypes.Tree([6, -1, 1, 1, 5, 3, 3, 1]))