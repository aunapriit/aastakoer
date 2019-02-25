# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 2019
@author: Priit

Retrieve data about Bernese Mountain dog dogshows results held in Estonia and write it to the csv file. 
Alter the year and breed info for other use case.
"""
import csv, time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

# change year value to get the results from a different year
year = '2018'
breed = 'BERNI ALPI KARJAKOER' # breed name to search from results
FCI_link = 'FCI2' # change according to FCI link
breed_lowercase = ''.join(breed.lower())
filename = 'dogshows_' + breed_lowercase + '_' + year + '.csv'
delay = 3

# check if xpath does work and return value or exception
def try_el(xpath):
    try:
        value = driver.find_element_by_xpath(xpath).text
    except NoSuchElementException:
        print ('xpath not working: ', xpath)
        value = "no value"
        pass
    return value

driver = webdriver.Firefox()
driver.get("https://online.kennelliit.ee/show/")
time.sleep(delay)

# write csv file header and define fieldnames
fieldnames = ['naitus', 'klass', 'koer', 'koeralink', 'tulemused', 'kohtunik']
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

#number = 0 #counter for testing

# uncomment to select table rows that contains links to all shows
#links = [link.get_attribute("href") for link in driver.find_elements_by_link_text('Tulemused')]

# select table rows that contains links to shows in certain year
linkscollection = driver.find_elements_by_xpath('//tr[contains(string(), "' + year +'")]//a[contains(string(), "Tulemused")]')
links = [link.get_attribute("href") for link in linkscollection]

# iteration over collected links to shows
for link in links:
    time.sleep(delay)
    
    driver.get(link) # get page of dogshow

    # heading of the dogshow
    driver.implicitly_wait(delay)
    naitus = driver.find_element_by_xpath('//*[@id="showname"]').text
    print(naitus)

    # get all results
    try:
        driver.implicitly_wait(delay)
        taistulemused = driver.find_element_by_link_text('Täistulemused')
        taistulemused.click()
    except NoSuchElementException:
        print ('Täistulemused link not found')
        pass # skip and continue
        
    # select class FCI2 results or skip it already there
    try:
        driver.implicitly_wait(delay)
        fci2 = driver.find_element_by_link_text(FCI_link)
        fci2.click()
    except NoSuchElementException:
        print ('FCI link not found')
        pass # just skip it and search for breed
        
    # click on breed results
    try:
        driver.implicitly_wait(delay)
        bern = driver.find_element_by_link_text(breed)
        bern.click()
    except NoSuchElementException:
        print ('Dog link not found')
        continue # no breed found - skip following code and begin with new dogshow
    
    # wait for table loading
    try:
        el = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
        print ("Table loaded!")
    except TimeoutException:
        print ("Loading took too much time!")
        pass
        
    # select all rows from the dogshow results and check if there is information about dogs
    try:
        driver.implicitly_wait(delay)
        tablerows = driver.find_elements_by_xpath('//*[@id="results"]/table//tr')
        if len(tablerows) < 2:
            print('1 row table')
            continue
    except NoSuchElementException:
        print('no table rows found')
        continue # no table found - skip following code and begin with new dogshow
    
    # information about the judge of the show
    kohtunik = driver.find_element_by_xpath('//*[@id="results"]/table//tr[1]/td').text    
    
    # iteration over table rows and write data to file
    for i in range(2, len(tablerows)+1):
        
        xpath_str = '//*[@id="results"]/table//tr[' + str(i) + ']/td'
        
        klass = try_el(xpath_str + '[1]') # doc class
        koer = try_el(xpath_str +'[3]') # dog name and code
        try:
            koeralink = driver.find_element_by_xpath(xpath_str +'[3]/a').get_attribute("href")# link to the dog data 
        except NoSuchElementException:
            print ('no dog link')
            koeralink = "no value"
            pass
        tulemused = try_el(xpath_str +'[4]') # dog show results
        
        # makerow of data and write it to the csv file
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'naitus': naitus, 'klass': klass, 'koer': koer, 'koeralink': koeralink, 'tulemused': tulemused, 'kohtunik': kohtunik})
            
    # use break and counter for testing 
    """if number == 5:
        break
    number = number + 1"""
    
driver.close()
