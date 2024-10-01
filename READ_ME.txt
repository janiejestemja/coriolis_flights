"Coriolis Flight Data Analysis – A Learning Journey in Data Science and C++ Integration."
Project Documentation

0. Project Structure Overview

Development Environment: Linux Mint 22 Cinnamon  

Tested with: 

Python 3.12.3
g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0

Versions of Python libraries in use:

numpy - 2.1.0 
pandas - 2.2.2
geopandas - 1.0.1 

matplotlib - 3.9.2 
seaborn - 0.13.2

shapely - 2.0.6
geodatasets - 2024.8.0

Text-Editor: 
Jupyter Notebook 7.2.1

Directories:

corilolis_module

    Contains:
        coriolis_module.cpp: Written using the Python C-API, contains the direction_vector function.
        setup.py: Python script to compile the .cpp file into a .so shared object file.

How to build:
Run the following command in your Linux terminal:

bash
>>>python setup.py build_ext --inplace

For virtual environments, copy the resulting .so file to:

bash
>>>path_to/venv/lib/python/site-packages/

datasets # not included in repository

    Contains:
        flights_sample_3m.csv: Main flight dataset.
        airports.csv: Contains geographical coordinates for airport IATA codes.
        flights_sample_3m.csv.zip: Compressed version of the flight dataset.
        note_flightsample.txt: Notes from Kaggle on the flight dataset.

Comment: During EDA, I figured out that the "IATA" codes in flights_sample_3m.csv correspond to primary keys for U.S. airports, giving me a hint to download airports.csv to match these codes with coordinates.

1. Introduction & Motivation

This project started as an experiment born from three simple (but strange) ideas:

    Download a random CSV file from the internet.
    Do an analysis nobody asked for—aim for irrelevant but fun results.
    Ignore the dataset documentation to draw conclusions directly from the data.

It grew into a hands-on experience combining Python, C++, and data science tools. Although the results might be insignificant, the project reflects my ongoing learning journey. By embracing confusion, I figured out how to work with larger datasets and even wrote a C++ module.

2. Technical Overview

Disclaimer: This project is primarily for learning and exploration in C++ and Python integration. The results are mostly for fun and to experiment with different techniques—use with caution!

C++ Module:

Why C++?
While Python handles data manipulation well, the drift calculations require heavy computations. This is where C++ comes in, offering better performance for those operations.

What It Does:
The module helps with the calculations of the Coriolis drift of planes, reflecting the effects of Earth's rotation on their paths.

Datasets:

    flights_sample_3m.csv: Flight data, including departure/arrival times and locations.
    airports.csv: Links IATA codes to geographical locations.

3. Learning Journey & Challenges

This project captures my learning curve in data science, coding, and integrating C++ with Python. Avoiding dataset documentation was a misstep, but ultimately, it allowed me to experiment and build from scratch. Some challenges I faced include:

    Large Data: Handling 3 million rows of flight data slowed down my computations. Switching to C++ improved performance, though the true gains remained insignificant.
    Documentation: The importance of good documentation became very clear when I had to reverse-engineer the dataset. Lesson learned!

4. Fun but Insignificant Results

One particularly amusing insight: Coriolis drift’s effect on airplanes—though calculable—is practically irrelevant. But hey, it was fun to explore!

5. Closing Thoughts

This project mirrors my nonlinear journey in computer science, a path filled with unexpected turns. The results might not change the world, but I’ve learned how to integrate Python and C++, navigate large datasets, and above all—document my work.


6. License
This project is licensed under the MIT License. See the LICENSE file for more details.