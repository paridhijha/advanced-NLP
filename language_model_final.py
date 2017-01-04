################################################################################################################################################
#########################################         ANLP ASSIGNMENT 1 LANGUAGE MODELLING         #################################################
################################################################################################################################################

from __future__ import division
from collections import defaultdict
from math import log
import random
import collections
import re
import os
import math
import sys
import operator
vocab = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',',','.',' ','0'}
vocab_size=30
log_base=2
# Assumption : the following 3 training files are present in local directory from where the program is run
file_training_en='training.en'
file_training_es='training.es'
file_training_de='training.de'

##############################################################################################################################################
# Function Name     : preprocess_line()
# Function Purpose  : removes all special characters (except , and .) ,converts uppercase to lowercase(all A-Z) and converts all digits(1-9) to 0
# Function Input    : string, var name is text
# Function Output   : modified string
# Other Remarks     : The function uses library re and its function sub to perform the string operations. The regex expression is a global variable

def preprocess_line(text):
        if not text:
            print ("Empty string in preprocess() function. Exiting...")
            sys.exit(1)
        text1 =  re.sub('[^a-z^A-Z^0-9 ,.]','', text.lower())
        return re.sub('[\d]','0', text1)

##############################################################################################################################################
# Function Name     : preprocess_file()
# Function Purpose  : to preprocess text in a file using the function preprocess_line() one single line at a time
# Function Input    : string which represents a filename
# Function Output   : string which represents a newly created filename with a prefix "preprocessed_" attached
# Other Remarks     : the function expects an existing filename, otherwise throws an error. The new file is created in local directory

def preprocess_file(filename):
        if not os.path.isfile(filename):
            print  ("The training file does not exist. Exiting")
            sys.exit()
        if  (os.stat(filename).st_size) <= 0:
            print  ("The training file is empty. Exiting")
            sys.exit()
        inputfile = open(filename,'r')
        outputfile= open("preprocessed_"+filename,'w')
        for line in inputfile:
            processed_line = preprocess_line(line)
            outputfile.write(processed_line)
        #outputfile.write('\n')
        inputfile.close(),outputfile.close()
        return "preprocessed_"+filename

##############################################################################################################################################
# Function Name     : count_trigrams()
# Function Purpose  : iterates over each character in a file, counts the total number of trigrams present in the
#                     file and returns a default dictionary with key=trigram and value=probability of that trigram
# Function Input    : string which represents a filename
# Function Output   : default dict object containing the counts of trigrams
# Other Remarks     : requires a valid file with atleast one trigram, otherwise throws an error

def count_trigrams(filename):
    file_size = os.stat(filename).st_size
    current_index = 1
    if  file_size <= 0:
        print  ("Error : The training file is empty. Exiting")
        Sys.exit()
    dict_count = defaultdict(int)
    with open(filename) as file:
        ch1 = file.read(1)
        ch1 = file.read(1)
        if ch1 =='':
            print ("Error : The training file is empty. Exiting")
            sys.exit()
        ch2 = file.read(1)
        if  ch2 =='':
            print ("Error : The training file has 0 character. Exiting")
            sys.exit()
        ch3 = file.read(1)
        if  ch3 == '':
            print ("Error : The training file has 1 characters. Exiting")
            sys.exit()
        while current_index< file_size-3:
            trigram = ch1+ch2+ch3
            dict_count[trigram]+=1
            ch1=ch2
            ch2=ch3
            ch3 = file.read(1)
            current_index+=1
    file.close()
    return dict_count

##############################################################################################################################################
# Function Name     : count_bigrams()
# Function Purpose  : iterates over each character in a file, counts the total number of bigrams present in the
#                     file and returns a default dictionary with key=bigram and value=probability of that bigram
# Function Input    : string which represents a filename
# Function Output   : default dict object containing the counts of bigrams
# Other Remarks     : requires a valid file with atleast one bigram, otherwise throws an error

def count_bigrams(filename):
    file_size = os.stat(filename).st_size
    current_index = 1
    if  file_size <= 0:
        print  ("\n Error : The training file is empty. Program aborted. \n")
        Sys.exit()
    dict_count = defaultdict(int)
    with open(filename) as file:
        ch1 = file.read(1)
        ch1 = file.read(1)
        if ch1 =='':
            print ("\n Error : The training file is empty. Program aborted. \n")
            sys.exit()
        ch2 = file.read(1)
        if  ch2 =='':
            print ("\n Error : The training file has 0 character. Program aborted. \n")
            sys.exit()
        while current_index< file_size-2:
            bigram = ch1+ch2
            dict_count[bigram]+=1
            ch1=ch2
            ch2 = file.read(1)
            current_index+=1
    file.close()
    return dict_count

#############################################################################################################################################
# Function Name     : count_unigrams()
# Function Purpose  : iterates over each character in a file, counts the total number of unigrams present in the
#                     file and returns a default dictionary with key=unigram and value=probability of that unigram
# Function Input    : string which represents a filename
# Function Output   : default dict object containing the counts of unigrams
# Other Remarks     : requires a valid file with atleast one unigram, otherwise throws an error

def count_unigrams(filename):
    file_size = os.stat(filename).st_size
    current_index = 1
    if  file_size <= 0:
        print  ("\n Error : The training file is empty. Program aborted. \n")
        Sys.exit()
    dict_count = defaultdict(int)
    with open(filename) as file:
        ch1 = file.read(1)
        ch1 = file.read(1)
        if ch1 =='':
            print ("\n Error : The training file is empty. Program aborted. \n")
            sys.exit()
        ch2 = file.read(1)
        if  ch2 =='':
            print ("\n Error : The training file has 0 character. Program aborted. \n")
            sys.exit()
        while current_index< file_size-1:
            unigram = ch1
            dict_count[unigram]+=1
            ch1 = file.read(1)
            current_index+=1
    file.close()
    return dict_count

##############################################################################################################################################
# Function Name     : get_prob_of_trigram()
# Function Purpose  : Expects a nested default dictionary and a trigram, using the last character of the given
#                     trigram as the key of outer dictionary it fetches its corresponding value. It expects this value
#                     to store an inner dictionary. It then uses the first two characters of trigram as the key and
#                     fetches the value of this inner dictionary. This value corresponds to the stored probability of
#                     trigram and is returned by this function
# Function Input    : A string (var name is trigram) and a default dictionary (var name is lang_model)
# Function Output   : Float value (which represents the probability of the trigram)
# Other Remarks     : If the input dictionary is not a default dictionary, it simply returns the value stored at key=trigram
def get_prob_of_trigram(trigram,lang_model):
    if not trigram:
        print ("Input trigram is an empty string in get_prob_of_trigram() function. Exiting...")
        sys.exit(1)
    if type(lang_model[trigram]) is defaultdict:
        inner_key=trigram[0]+trigram[1]
        outer_key=trigram[2]
        inner_dict=lang_model[outer_key]
        #print "\nDebug : For trigram = ",trigram
        #print "Debug : Outer key is",outer_key, "Inner key is ",inner_key, "inner value is ",inner_dict[inner_key]
        return inner_dict[inner_key]
    return lang_model[trigram]

##############################################################################################################################################
# Function Name     : calc_perplexity_test_data()
# Function Purpose  : to calculate the perplexity given a language model and a test data file
# Function Input    : a string which stores test data and a dictionary which stores the trigrams and their respective probabilities
# Function Output   : a float value which is the calculated perplexity
# Other Remarks     : it calls function get_prob_of_trigram() internally to  calculate fetch the probability of a particular trigram.
#                     the probabilites are stored in log form and the calculated perplexity is converted back using math.pow() fn

def calc_perplexity_test_data(test_data,log_prob_dict):
    
    if not log_prob_dict:
        print ("\n Error : The probability dictionary is empty. Program aborted. \n")
        sys.exit()
    if not test_data:
        print ("\n Error : Empty string. Cannot calculate perplexity. Program aborted. \n")
        sys.exit()
    text=test_data[0]
    ch1=text[0]
    ch2=text[1]
    num_trigrams = len(text)-2.0
    log_prob_sum=0
    for index in range(2,len(text)):
        ch1= text[index-2]
        ch2= text[index-1]
        ch3= text[index]
        trigram = ch1+ch2+ch3
        current_log_prob=get_prob_of_trigram(trigram,log_prob_dict)
        log_prob_sum +=current_log_prob
    avg_log_prob_sum=log_prob_sum/num_trigrams
    exponent=-(avg_log_prob_sum)
    perplexity= math.pow(log_base,exponent)
    return perplexity

##############################################################################################################################################
# Function Name     : calc_perplexity_text()
# Function Purpose  : to calculate the perplexity given a language model and a text/string
# Function Input    : a string which stores test data and a dictionary which stores the trigrams and their respective probabilities
# Function Output   : a float value which is the calculated perplexity
# Other Remarks     : this function is a variation of calc_perplexity_test_data(), it expects a string instead of a file for its test data.

def calc_perplexity_text(text,log_prob_dict):
    
    if not log_prob_dict:
        print ("\n Error : The probability dictionary is empty. Program aborted. \n")
        sys.exit()
    if not text:
        print ("\n Error : Empty string. Cannot calculate perplexity. Program aborted. \n")
        sys.exit()
    ch1=text[0]
    print ch1
    ch2=text[1]
    num_trigrams = len(text)-2.0
    log_prob_sum=0
    for index in range(2,len(text)):
        ch1= text[index-2]
        ch2= text[index-1]
        ch3= text[index]
        trigram = ch1+ch2+ch3
        current_log_prob=get_prob_of_trigram(trigram,log_prob_dict)
        log_prob_sum +=current_log_prob
    avg_log_prob_sum=log_prob_sum/num_trigrams
    exponent=-(avg_log_prob_sum)
    perplex= math.pow(log_base,exponent)
    return perplex

##############################################################################################################################################
# Function Name     : estimate_log_probabilities_with_add1()
# Function Purpose  : calculates the probabilites of each trigram using Markov Model(using the previous bigram count and trigram count). it
#                     also applies add 1 smoothing (laplace smoothing) so that none of the probabilities remain zero
# Function Input    : two dictionaries storing the bigram counts and trigram counts of the training file
# Function Output   : one nested dictionary which holds the calculated probabilities for each possible trigram from training data
# Other Remarks     : all probabilities are stored in log form. the returned dictionary is a nested dictionary where its outer key is the
#                     3rd character of the trigram and outer value is the link to inner dictionary. each nested inner dictionary has inner key
#                     the bigram(1st and 2nd character of the trigram) and value as the calculated probability in log form. together the outer
#                     and inner key represent the trigram and store their corresponding log probabilites.

def estimate_log_probabilities_with_add1(raw_bigram_count, raw_trigram_count):

    if not raw_bigram_count:
        print ("\n Error : The raw_bigram_count dictionary is empty. Program aborted. \n")
        sys.exit()
    if not raw_trigram_count:
        print ("\n Error : The raw_trigram_count dictionary is empty. Program aborted. \n")
        sys.exit()
    # prob_dict is a nested default dictionary
    prob_dict = defaultdict(lambda: defaultdict(float))
    for outer_key in vocab:
        inner_dict = defaultdict(float)
        for inner_key1 in vocab:
            for inner_key2 in vocab:
                bigram=inner_key1+inner_key2
                trigram=inner_key1+inner_key2+outer_key
                trigram_count = raw_trigram_count[trigram]
                bigram_count  = raw_bigram_count[bigram]
                est_prob_value = (trigram_count+1)/(bigram_count+vocab_size)
                if not est_prob_value:
                    print ("\n Error : The estimated probability cannot be zero. It will result in Math Domain Error. Program aborted. \n")
                    sys.exit()
                log_est_prob_value = math.log(est_prob_value,log_base)
                inner_dict[bigram] = log_est_prob_value
        prob_dict[outer_key]=inner_dict
    return prob_dict

##############################################################################################################################################
# Function Name     : convert_to_log_probabilities()
# Function Purpose  : convert (float) probabilites from a dictionary into its log form and stores it in another dictionary
# Function Input    : dictionary (var = prob_dict)
# Function Output   : dictionary (var = log_prob_dict)
# Other Remarks     : it uses global variable log_base and maths function log for log conversion

def convert_to_log_probabilities(prob_dict):

    if not prob_dict:
        print ("\n Error : The probability dictionary is empty. Program aborted. \n")
        sys.exit()

    log_prob_dict=defaultdict(float)
    for trigram,prob_value in prob_dict.iteritems():
        if prob_value:
            log_prob_value= math.log(prob_value,log_base)
            log_prob_dict[trigram]=log_prob_value
    return log_prob_dict

##############################################################################################################################################
# Function Name     : print_estimated_log_prob()
# Function Purpose  : to print a given dictionary with trigrams and its log probabilities to a file
# Function Input    : dictionary (var = dict), string value (var = lang) for English, Spanish or German
# Function Output   : none
# Other Remarks     : a file with name log_model_prob_*.txt gets generated in local directory

def print_estimated_log_prob(dict, lang):

    if not dict:
        print ("\n Error : The probability dictionary is empty. Program aborted. \n")
        sys.exit()
    filename = "log_model_prob_"+lang+".txt"
    outputfile= open(filename,'w')
    outputfile.write('\n')
    outputfile.write("   Trigram       Log Probability")
    for outer_key,inner_dict in dict.iteritems():
        for inner_key,log_prob_value in inner_dict.iteritems():
            prob_value=math.pow(log_base,-log_prob_value)
            line="  "+inner_key+outer_key+"     "+str(prob_value)+"\n"
            outputfile.write(line)
    outputfile.close()
    return

##############################################################################################################################################
# Function Name     : sort_estimated_log_prob()
# Function Purpose  : to print the sorted trigrams and its log probabilities to a file
# Function Input    : dictionary (var = dict), string value (var = lang) for English, Spanish or German
# Function Output   : none
# Other Remarks     : a file with name sorted_model_prob_*.txt gets generated in local directory

def sort_estimated_log_prob(dict, lang):

    if not dict:
        print ("\n Error : The probability dictionary is empty. Program aborted. \n")
        sys.exit()
    prob_dict= {}
    sorted_prob_dict = {}
    filename = "sorted_model_prob_"+lang+".txt"
    outputfile= open(filename,'w')
    outputfile.write('\n')
    outputfile.write("   Trigram       Sorted Probability")
    for outer_key,inner_dict in dict.iteritems():
        for inner_key,log_prob_value in inner_dict.iteritems():
            prob_value=math.pow(log_base,-log_prob_value)
            trigram=outer_key+inner_key
            prob_dict[trigram]=prob_value
    sorted_prob_dict = sorted(prob_dict.items(), key=operator.itemgetter(1))
    for key,value in sorted_prob_dict:
            line="  "+key+"     "+str(value)+"\n"
            outputfile.write(line)
    outputfile.close()
    return

##############################################################################################################################################
# Function Name     : get_history_english()
# Function Purpose  : writes the probabilities of all history of given bigram to a file (for language model english)
# Function Input    : string which stores the bigram
# Function Output   : none to screen, it creates a new file history_*.txt in the local directory
# Other Remarks     : this is the solution for subpart of Q 3.3.3 which asks to calculate 2 ch history of th. this fn is called from the main
#                     with parameter bigram='th'

def get_history_english(bigram):
    
    preprocessed_file_en = preprocess_file(file_training_en)
    trigram_count_dict_en = count_trigrams(preprocessed_file_en)
    bigram_count_dict_en = count_bigrams(preprocessed_file_en)
    
    est_log_prob_dict_en= estimate_log_probabilities_with_add1(bigram_count_dict_en,trigram_count_dict_en)
    outputfile= open("history_"+bigram+".txt",'w')
    
    outputfile.write('\n')
    outputfile.write("Trigram     Probability\n")
    
    for ch in vocab:
        trigram = bigram+ch
        log_prob_value = get_prob_of_trigram(trigram,est_log_prob_dict_en)
        prob_value = math.pow(log_base,(log_prob_value))
        line="  "+trigram+"            "+str(prob_value)+"\n"
        outputfile.write(line)
    outputfile.write('\n')
    outputfile.close()
    return

##############################################################################################################################################
# Function Name     : generate_random()
# Function Purpose  : to generate random output(300 characters) for each of the 3 languages using the 3 lang models trained before
# Function Input    : none
# Function Output   : outputs 3 new files in local directory with name random_generated_text_*.txt
# Other Remarks     : this is solution for Q 3.3.4, it calls 2 helper functions get_trigram_with_highest_prob() and
#                     get_trigram_with_highest_prob_given_bigram() to get the next random character with higest trigram probability

def generate_random():
    
    num_chars=300
    count=0
    
    preprocessed_file_en = preprocess_file(file_training_en)
    preprocessed_file_es = preprocess_file(file_training_es)
    preprocessed_file_de = preprocess_file(file_training_de)
    
    trigram_count_dict_en = count_trigrams(preprocessed_file_en)
    trigram_count_dict_es = count_trigrams(preprocessed_file_es)
    trigram_count_dict_de = count_trigrams(preprocessed_file_de)
    
    bigram_count_dict_en = count_bigrams(preprocessed_file_en)
    bigram_count_dict_es = count_bigrams(preprocessed_file_es)
    bigram_count_dict_de = count_bigrams(preprocessed_file_de)
    
    outputfile_en = open("random_generated_text_english.txt",'w')
    outputfile_es = open("random_generated_text_spanish.txt",'w')
    outputfile_de = open("random_generated_text_german.txt",'w')
    
    est_log_prob_dict_en = estimate_log_probabilities_with_add1(bigram_count_dict_en,trigram_count_dict_en)
    est_log_prob_dict_es = estimate_log_probabilities_with_add1(bigram_count_dict_es,trigram_count_dict_es)
    est_log_prob_dict_de = estimate_log_probabilities_with_add1(bigram_count_dict_de,trigram_count_dict_de)
    
    starting_trigram_en = get_trigram_with_highest_prob(est_log_prob_dict_en)
    starting_trigram_es = get_trigram_with_highest_prob(est_log_prob_dict_es)
    starting_trigram_de = get_trigram_with_highest_prob(est_log_prob_dict_de)
    
    outputfile_en.write(starting_trigram_en)
    outputfile_es.write(starting_trigram_es)
    outputfile_de.write(starting_trigram_de)
    
    current_bigram_en = starting_trigram_en[1] + starting_trigram_en[2]
    current_bigram_es = starting_trigram_es[1] + starting_trigram_es[2]
    current_bigram_de = starting_trigram_de[1] + starting_trigram_de[2]
    
    while (count < num_chars):
        
        current_trigram_en = get_trigram_with_highest_prob_given_bigram(current_bigram_en , est_log_prob_dict_en)
        current_trigram_es = get_trigram_with_highest_prob_given_bigram(current_bigram_es , est_log_prob_dict_es)
        current_trigram_de = get_trigram_with_highest_prob_given_bigram(current_bigram_de , est_log_prob_dict_de)
        
        current_bigram_en = current_trigram_en[1] + current_trigram_en[2]
        current_bigram_es = current_trigram_es[1] + current_trigram_es[2]
        current_bigram_de = current_trigram_de[1] + current_trigram_de[2]
        
        outputfile_en.write(current_trigram_en[2])
        outputfile_es.write(current_trigram_es[2])
        outputfile_de.write(current_trigram_de[2])
        
        count+=1
    
    outputfile_en.write("\n")
    outputfile_es.write("\n")
    outputfile_de.write("\n")

    outputfile_en.close()
    outputfile_es.close()
    outputfile_de.close()
    
    return

def get_trigram_with_highest_prob(prob_dict):
    
    bigram=". "
    highest_prob_value=0.0
    
    for outer_key,inner_dict in prob_dict.iteritems():
        inner_key = bigram
        current_prob_value = math.pow(log_base,inner_dict[inner_key])
        if ord(outer_key)!=48:
            if (highest_prob_value < current_prob_value):
                highest_prob_value=current_prob_value
                highest_prob_trigram=bigram+outer_key
    return highest_prob_trigram

def get_trigram_with_highest_prob_given_bigram(bigram,prob_dict):
    
    highest_prob_trigram=""
    highest_prob_value=0.0
    for outer_key,inner_dict in prob_dict.iteritems():
        inner_key = bigram
        current_prob_value = math.pow(log_base,inner_dict[inner_key])
        if highest_prob_value < current_prob_value:
            highest_prob_value=current_prob_value
            highest_prob_trigram=bigram+outer_key
    return highest_prob_trigram

###########################################################################################################################################
# Function Name     : detect_lang_test_doc()
# Function Purpose  : to detect the language given a test doc and print the results to a result file
# Function Input    : a test document containing the text for one among the 3 languages. the file is expected to be in the local dir
# Function Output   : a results file - perplexity_result_test_file.txt which is created in the local dir
# Other Remarks     : it also prints the perplexity of all 3 models for the given test data. this fn is the solution for subpart of Q 3.3.3

def detect_lang_test_doc(test_doc):
    
    newfile = preprocess_file(test_doc)
    with open (newfile, "r") as test_file:
        text=test_file.readlines()
    
    est_log_prob_dict_en = generate_lang_model_en()
    est_log_prob_dict_es = generate_lang_model_es()
    est_log_prob_dict_de = generate_lang_model_de()
    
    perplex_en = calc_perplexity_test_data(text,est_log_prob_dict_en)
    perplex_es = calc_perplexity_test_data(text,est_log_prob_dict_es)
    perplex_de = calc_perplexity_test_data(text,est_log_prob_dict_de)
    
    if (perplex_en < perplex_es):
        if (perplex_en < perplex_de):
            lang="English"
        else:
            lang="German"
    else:
        if (perplex_es < perplex_de):
            lang="Spanish"
        else:
            lang="German"

    outputfile= open("perplexity_result_test_file.txt",'w')
    outputfile.write('\n')
    
    line0 = "Question 3.3.5 Following are the results of perplexities calculated for each of the three langugage models - english, spanish and german \n"
    line1 = "The perplexity for english for test file is "+str(perplex_en)+"\n"
    line2 = "The perplexity for spanish for test file is "+str(perplex_es)+"\n"
    line3 = "The perplexity for german  for test file is "+str(perplex_de)+"\n"
    line4 = "\n The detected language for test file is "+lang+"\n"
    
    outputfile.write( "\n"  )
    outputfile.write( line0 )
    outputfile.write( "\n"  )
    
    outputfile.write( line1 )
    outputfile.write( line2 )
    outputfile.write( line3 )
    outputfile.write( "\n"  )
    outputfile.write( line4 )
    outputfile.write( "\n"  )
    
    return

###########################################################################################################################################
# Function Name     : detect_lang_for_text()
# Function Purpose  : to detect the language given a text and print the results to a result file
# Function Input    : the text for one among the 3 languages. the file is expected to be in the local dir
# Function Output   : a results file - perplexity_result_test_file.txt which is created in the local dir
# Other Remarks     : this is a variation of fn detect_lang_test_doc(). instead of a test file, it inputs a text string.

def detect_lang_for_text(text):
    
    est_log_prob_dict_en = generate_lang_model_en()
    est_log_prob_dict_es = generate_lang_model_es()
    est_log_prob_dict_de = generate_lang_model_de()
    
    perplex_en = calc_perplexity_text(text,est_log_prob_dict_en)
    perplex_es = calc_perplexity_text(text,est_log_prob_dict_es)
    perplex_de = calc_perplexity_text(text,est_log_prob_dict_de)
    
    if (perplex_en <= perplex_es):
        if (perplex_en <= perplex_de):
            lang="English"
        else:
            lang="German"
    else:
        if (perplex_es <= perplex_de):
            lang="Spanish"
        else:
            lang="German"

    outputfile= open("perplexity_result_text.txt",'w')
    outputfile.write('\n')
    
    line0 = "Question 3.3.5 Following are the results of perplexities calculated for each of the three langugage models - english, spanish and german \n"
    line1 = "The perplexity for english for text = "+text+" is "+str(perplex_en)+"\n"
    line2 = "The perplexity for spanish for text = "+text+" is "+str(perplex_es)+"\n"
    line3 = "The perplexity for german  for text = "+text+" is "+str(perplex_de)+"\n"
    line4 = "\n The detected language for text = "+text+ " is  "+lang+"\n"
    
    outputfile.write( "\n"  )
    outputfile.write( line0 )
    outputfile.write( "\n"  )
    
    outputfile.write( line1 )
    outputfile.write( line2 )
    outputfile.write( line3 )
    outputfile.write( "\n"  )
    outputfile.write( line4 )
    outputfile.write( "\n"  )
    
    return lang

##############################################################################################################################################
# Function Name     : generate_lang_model_en() , generate_lang_model_es() , generate_lang_model_de()
# Function Purpose  : these 3 functions create the 3 language models
# Function Input    : the training file stored in the local directory
# Function Output   : return back nested dictionary which stores the trigram and its probabilities to the calling program, it also creates 3
#                     new files which represent the lang models and stores them in local directory using print_estimated_log_prob() fn
# Other Remarks     : this is part of the solution for Q 3.3.3

def generate_lang_model_en():
    preprocessed_file_en = preprocess_file(file_training_en)
    trigram_count_dict_en = count_trigrams(preprocessed_file_en)
    bigram_count_dict_en = count_bigrams(preprocessed_file_en)
    est_log_prob_dict_en= estimate_log_probabilities_with_add1(bigram_count_dict_en,trigram_count_dict_en)
    print_estimated_log_prob(est_log_prob_dict_en,"english")
    sort_estimated_log_prob(est_log_prob_dict_en,"english")
    return est_log_prob_dict_en

def generate_lang_model_es():
    preprocessed_file_es = preprocess_file(file_training_es)
    trigram_count_dict_es = count_trigrams(preprocessed_file_es)
    bigram_count_dict_es = count_bigrams(preprocessed_file_es)
    est_log_prob_dict_es= estimate_log_probabilities_with_add1(bigram_count_dict_es,trigram_count_dict_es)
    print_estimated_log_prob(est_log_prob_dict_es,"spanish")
    return est_log_prob_dict_es

def generate_lang_model_de():
    preprocessed_file_de = preprocess_file(file_training_de)
    trigram_count_dict_de = count_trigrams(preprocessed_file_de)
    bigram_count_dict_de = count_bigrams(preprocessed_file_de)
    est_log_prob_dict_de= estimate_log_probabilities_with_add1(bigram_count_dict_de,trigram_count_dict_de)
    print_estimated_log_prob(est_log_prob_dict_de,"german")
    return est_log_prob_dict_de

##############################################################################################################################################
######################################################## Main Body of the program ############################################################

#test_doc="test"
#detect_lang_test_doc(test_doc)
#generate_random()
#get_history_english("th")

###############################################################################################################################################
# Comment   Q3.1 Example - Find perplexity of [[abaab] when the probabilities of its trigram is given as
#prob_dict= {'[[a' : 0.2,
#            '[[b' : 0.8,
#            '[[]' : 0.0,
#            '[aa' : 0.2,
#            '[ab' : 0.7,
#            '][a' : 0.1,
#            '[ba' : 0.15,
#            '[bb' : 0.75,
#            '[b]' : 0.1,
#            'aaa' : 0.4,
#            'aab' : 0.5,
#            'aa]' : 0.1,
#            'aba' : 0.6,
#            'abb' : 0.3,
#            'ab]' : 0.1,
#            'baa' : 0.25,
#            'bab' : 0.65,
#            'ba]' : 0.1,
#            'bba' : 0.5,
#            'bbb' : 0.4,
#            'bb]' : 0.1
#             }
#print "\n"
#print "\n Comment : Question 3.3.1 The perplexity calculated by hand for text='[[abaab]' is 3.14 and for text='[ba]' is 8.16 The test case has following trigram probabilities."
##print "\n Trigram      Probability"
##print prob_dict
#text1='[[abaab]'
#print "Comment : Perplexity for text = ",text1, "is",calc_perplexity_text(text1,prob_dict)
#text2= '[ba]'
#print "Comment : Perplexity for text = ",text2, "is",calc_perplexity_text(text2,prob_dict)
## Converting all probabilities in their log form
#log_prob_dict = convert_to_log_probabilities(prob_dict)
#
##print "Debug : \n Trigram     Log Probability"
##print prob_dict
#text3='[[abaab]'
#print "Comment : Perplexity  using log probabilities for text = ",text3, "is",calc_perplexity_text(text3,log_prob_dict)
#text4= '[ba]'
#print "Comment : Perplexity  using log probabilities for text = ",text4, "is",calc_perplexity_text(text4,log_prob_dict)

################################################################################################################################################
########################################################        END    PROGRAM        ##########################################################
################################################################################################################################################



















