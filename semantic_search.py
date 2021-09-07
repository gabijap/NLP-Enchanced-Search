"""
    ***************************************************************************************
    *    This is simple demonstration how to search.
    *    Sentences from the following Wikipedia articles are used:
    *    https://en.wikipedia.org/wiki/Colonization_of_Mars
    *    https://en.wikipedia.org/wiki/Roman_Empire
    ***************************************************************************************
"""

import sys

import numpy as np
import torch

from quick_thoughts import QT, load_models
from utils import cosine_similarity

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)

CHECK_POINT_F = '2021_04_05_18_36_23'
CHECK_POINT_G = '2021_04_06_11_41_59'

encoder_f, encoder_g = load_models(CHECK_POINT_F, CHECK_POINT_G)
embedder = QT(encoder_f, encoder_g)

sentences1 = [
    "An efficient framework for learning sentence representations .",
    "efficient framework for learning sentence representations .",
    "framework for learning sentence representations .",
    "for learning sentence representations .",
    "learning sentence representations .",
    "sentence representations .",
    "representations .",
    "."
]

sentences2 = [
    "An efficient framework for learning sentence representations .",
    "An efficient framework for learning sentence representations",
    "An efficient framework for learning sentence",
    "An efficient framework for learning",
    "An efficient framework for",
    "An efficient framework",
    "An efficient",
    "An"
]

sentences3 = [
    "An efficient framework for learning sentence representations .",
    "An efficient framework for studying sentence representations .",
    "An efficient framework for learning text representations .",
    "An coherent framework for learning sentence representations .",
    "An efficient foundation for learning sentence expressions .",
    "An efficient method for learning sentence embeddings .",
    "An efficient method for learning sentence vector presentations .",
    "An coherent method for training passage vector expression .",
    "efficient method for learning sentence embeddings ."
]

sentences4 = [
    "semantic relations between keywords from scientific articles",
    "semantic relations between keywords from research articles",
    "semantic relations between tokens from scientific articles",
    "semantic relations between tokens from scientific papers",
    "semantic relations between tokens from scientific documents",
    "semantic relations between tokens from research documents",
    "as a starting point, semantic relations between keywords from scientific articles",
    "as a starting point, semantic relations between keywords from scientific articles could be extracted in order to classify",
    "as a starting point, semantic relations between keywords from scientific articles could be extracted in order to classify articles",
    "Towards a Semantic Search Engine for Scientific Articles"
]

sentences5 = [
    "Learn to walk before you run",
    "Learn basic skills first before venturing into complex things",

    "Discretion is the better part of valor",
    "It is wise to be careful and not show unnecessary bravery",

    "A picture is worth a thousand words",
    "An image can tell a story better than words",

    "There is no place like home",
    "Home is the most comfortable place in the world"
]

sentences = [
    'The colonization of Mars by humans is an ongoing debate. ',
    'Some people want to colonize the planet Mars. ',
    'Satellite imagery shows that there is frozen ground water on the planet. ',
    'Mars also has a thin atmosphere. ',
    'Because of this, it has potential to host humans and other organic life. ',
    'That makes Mars the best choice for a thriving colony off the Earth. ',
    'The Moon has also been proposed as the first location for human colonization but it not known to have air or water.'
]

sentences7 = [
    'The difference in gravity would negatively affect human health by weakening bones and muscles. ',
    'There is also risk of osteoporosis and cardiovascular problems. ',
    'Current rotations on the International Space Station put astronauts in zero gravity for six months, a comparable length of time to a one-way trip to Mars. ',
    'This gives researchers the ability to better understand the physical state that astronauts going to Mars would arrive in. ',
    'Once on Mars, surface gravity is only 38% of that on Earth. ',
    'Microgravity affects the cardiovascular, musculoskeletal and neurovestibular (central nervous) systems. ',
    'The cardiovascular effects are complex. ',
    'On earth, blood within the body stays 70% below the heart, and in microgravity this is not case due to nothing pulling the blood down. ',
    'This can have several negative effects. Once entering into microgravity, the blood pressure in the lower body and legs is significantly reduced.',
    'This causes legs to become weak through loss of muscle and bone mass. ',
    'Astronauts show signs of a puffy face and chicken legs syndrome. ',
    'After the first day of reentry back to earth, blood samples showed a 17% loss of blood plasma, which contributed to a decline of erythropoietin secretion.',
    "On the skeletal system which is important to support our body's posture, long space flight and exposure to microgravity cause demineralization and atrophy of muscles. ",
    "During re-acclimation, astronauts were observed to have a myriad of symptoms including cold sweats, nausea, vomiting and motion sickness.",
    "Returning astronauts also felt disorientated. ",
    "Journeys to and from Mars being six months is the average time spent at the ISS. ",
    "Once on Mars with its lesser surface gravity (38% percent of Earth's), these health effects would be a serious concern.",
    "Upon return to Earth, recovery from bone loss and atrophy is a long process and the effects of microgravity may never fully reverse."
]

sentences8 = [
    'The difference in gravity would negatively affect human health by weakening bones and muscles. ',
    'There is also risk of osteoporosis and cardiovascular problems. ',
    'Current rotations on the International Space Station put astronauts in zero gravity for six months, a comparable length of time to a one-way trip to Mars. ',
    'This gives researchers the ability to better understand the physical state that astronauts going to Mars would arrive in. ',
    'Once on Mars, surface gravity is only 38% of that on Earth. ',
    'Microgravity affects the cardiovascular, musculoskeletal and neurovestibular (central nervous) systems. ',
    'The cardiovascular effects are complex. ',
    'On earth, blood within the body stays 70% below the heart, and in microgravity this is not case due to nothing pulling the blood down. ',
    'This can have several negative effects. Once entering into microgravity, the blood pressure in the lower body and legs is significantly reduced.',
    'This causes legs to become weak through loss of muscle and bone mass. ',
    'Astronauts show signs of a puffy face and chicken legs syndrome. ',
    'After the first day of reentry back to earth, blood samples showed a 17% loss of blood plasma, which contributed to a decline of erythropoietin secretion.',
    "On the skeletal system which is important to support our body's posture, long space flight and exposure to microgravity cause demineralization and atrophy of muscles. ",
    "During re-acclimation, astronauts were observed to have a myriad of symptoms including cold sweats, nausea, vomiting and motion sickness.",
    "Returning astronauts also felt disorientated. ",
    "Journeys to and from Mars being six months is the average time spent at the ISS. ",
    "Once on Mars with its lesser surface gravity (38% percent of Earth's), these health effects would be a serious concern.",
    "Upon return to Earth, recovery from bone loss and atrophy is a long process and the effects of microgravity may never fully reverse.",

    "Taxation under the Empire amounted to about 5% of the Empire's gross product.",
    "The typical tax rate paid by individuals ranged from 2 to 5%.",
    'The tax code was "bewildering" in its complicated system of direct and indirect taxes, some paid in cash and some in kind.',
    "Taxes might be specific to a province, or kinds of properties such as fisheries or salt evaporation ponds; they might be in effect for a limited time.",
    "Tax collection was justified by the need to maintain the military, and taxpayers sometimes got a refund if the army captured a surplus of booty.",
    "In-kind taxes were accepted from less-monetized areas, particularly those who could supply grain or goods to army camps.",
    "Personification of the River Nile and his children, from the Temple of Serapis and Isis in Rome (1st century AD)",
    "The primary source of direct tax revenue was individuals, who paid a poll tax and a tax on their land, construed as a tax on its produce or productive capacity.",
    "Supplemental forms could be filed by those eligible for certain exemptions; for example, Egyptian farmers could register fields as fallow and tax-exempt depending on flood patterns of the Nile.",
    "Tax obligations were determined by the census, which required each head of household to appear before the presiding official and provide a headcount of his household, as well as an accounting of property he owned that was suitable for agriculture or habitation.",
    "A major source of indirect-tax revenue was the portoria, customs and tolls on imports and exports, including among provinces.",
    "Special taxes were levied on the slave trade.",
    "Towards the end of his reign, Augustus instituted a 4% tax on the sale of slaves, which Nero shifted from the purchaser to the dealers, who responded by raising their prices.",
    'An owner who manumitted a slave paid a "freedom tax", calculated at 5% of value.',
    'An inheritance tax of 5% was assessed when Roman citizens above a certain net worth left property to anyone but members of their immediate family.',
    "Revenues from the estate tax and from a 1% sales tax on auctions went towards the veterans' pension fund (aerarium militare).",
    "Low taxes helped the Roman aristocracy increase their wealth, which equalled or exceeded the revenues of the central government.",
    'An emperor sometimes replenished his treasury by confiscating the estates of the "super-rich", but in the later period, the resistance of the wealthy to paying taxes was one of the factors contributing to the collapse of the Empire'
]

sentences9 = [
    "Conditions for human habitation",
    "Conditions on the surface of Mars are closer to the conditions on Earth in terms of temperature and sunlight than on any other planet or moon, except for the cloud tops of Venus.",
    "However, the surface is not hospitable to humans or most known life forms due to the radiation, greatly reduced air pressure, and an atmosphere with only 0.16% oxygen.",
    "In 2012, it was reported that some lichen and cyanobacteria survived and showed remarkable adaptation capacity for photosynthesis after 34 days in simulated Martian conditions in the Mars Simulation Laboratory.",
    "Some scientists think that cyanobacteria could play a role in the development of self-sustainable crewed outposts on Mars.",
    "They propose that cyanobacteria could be used directly for various applications, including the production of food, fuel and oxygen.",
    "Also products from their culture could support the growth of other organisms, opening the way to a wide range of life-support biological processes based on Martian resources.",
    "Humans have explored parts of Earth that match some conditions on Mars.",
    "Based on NASA rover data, temperatures on Mars (at low latitudes) are similar to those in Antarctica.",
    "The atmospheric pressure at the highest altitudes reached by piloted balloon ascents (35 km (114,000 feet) in 1961,[38] 38 km in 2012) is similar to that on the surface of Mars."
    "However, the pilots were not exposed to the extremely low pressure, as it would have killed them, but seated in a pressurized capsule.",
    "Human survival on Mars would require living in artificial Mars habitats with complex life-support systems.",
    "One key aspect of this would be water processing systems. "
    "Being made mainly of water, a human being would die in a matter of days without it.",
    "Even a 5–8% decrease in total body water causes fatigue and dizziness and a 10% decrease physical and mental impairment (See Dehydration).",
    "A person in the UK uses 70–140 litres of water per day on average.",
    "Through experience and training, astronauts on the ISS have shown it is possible to use far less, and that around 70% of what is used can be recycled using the ISS water recovery systems.",
    "Half of all water is used during showers.",
    "Similar systems would be needed on Mars, but would need to be much more efficient, since regular robotic deliveries of water to Mars would be prohibitively expensive.",
    "Potential access to in-situ water (frozen or otherwise) via drilling has been investigated by NASA.",
    "Effects on human health",
    "Mars presents a hostile environment for human habitation.",
    "Different technologies have been developed to assist long-term space exploration and may be adapted for habitation on Mars.",
    "The existing record for the longest consecutive space flight is 438 days by cosmonaut Valeri Polyakov, and the most accrued time in space is 878 days by Gennady Padalka.",
    "The longest time spent outside the protection of the Earth's Van Allen radiation belt is about 12 days for the Apollo 17 moon landing.",
    "This is minor in comparison to the 1100-day journey[45] planned by NASA as soon as the year 2028.",
    "Scientists have also hypothesized that many different biological functions can be negatively affected by the environment of Mars colonies.",
    "Due to higher levels of radiation, there are a multitude of physical side-effects that must be mitigated.",
    "In addition, Martian soil contains high levels of toxins which are hazardous to human health.",
    "Physical effects.",
    "The difference in gravity would negatively affect human health by weakening bones and muscles.",
    "There is also risk of osteoporosis and cardiovascular problems.",
    "Current rotations on the International Space Station put astronauts in zero gravity for six months, a comparable length of time to a one-way trip to Mars.",
    "This gives researchers the ability to better understand the physical state that astronauts going to Mars would arrive in.",
    "Once on Mars, surface gravity is only 38% of that on Earth.",
    "Microgravity affects the cardiovascular, musculoskeletal and neurovestibular (central nervous) systems.",
    "The cardiovascular effects are complex.",
    "On earth, blood within the body stays 70% below the heart, and in microgravity this is not case due to nothing pulling the blood down.",
    "This can have several negative effects.",
    "Once entering into microgravity, the blood pressure in the lower body and legs is significantly reduced.",
    "This causes legs to become weak through loss of muscle and bone mass.",
    "Astronauts show signs of a puffy face and chicken legs syndrome.",
    "After the first day of reentry back to earth, blood samples showed a 17% loss of blood plasma, which contributed to a decline of erythropoietin secretion.",
    "On the skeletal system which is important to support our body's posture, long space flight and exposure to microgravity cause demineralization and atrophy of muscles.",
    "During re-acclimation, astronauts were observed to have a myriad of symptoms including cold sweats, nausea, vomiting and motion sickness.",
    "Returning astronauts also felt disorientated.",
    "Journeys to and from Mars being six months is the average time spent at the ISS.",
    "Once on Mars with its lesser surface gravity (38% percent of Earth's), these health effects would be a serious concern.",
    "Upon return to Earth, recovery from bone loss and atrophy is a long process and the effects of microgravity may never fully reverse.",
    "Radiation",
    "Mars has a weaker global magnetosphere than Earth does as it has lost its inner dynamo."
    "This which significantly weakened the magnetosphere—the cause of so much radiation reaching the surface, despite its far distance from the Sun compared to Earth.",
    "Combined with a thin atmosphere, this permits a significant amount of ionizing radiation to reach the Martian surface.",
    "There are two main types of radiation risks to traveling outside the protection of Earth's atmosphere and magnetosphere: galactic cosmic rays (GCR) and solar energetic particles (SEP).",
    "Earth's magnetosphere protects from charged particles from the Sun, and the atmosphere protects against uncharged and highly energetic GCRs. "
    "There are ways to mitigate against solar radiation, but without much of an atmosphere, the only solution to the GCR flux is heavy shielding amounting to roughly 15 centimeters of steel, 1 meter of rock, or 3 meters of water, limiting human colonists to living underground most of the time.",
    "The Mars Odyssey spacecraft carries an instrument, the Mars Radiation Environment Experiment (MARIE), to measure the radiation.",
    "MARIE found that radiation levels in orbit above Mars are 2.5 times higher than at the International Space Station.",
    "The average daily dose was about 220 mGy (22 mrad)—equivalent to 0.08 Gy per year.",
    "A three-year exposure to such levels would exceed the safety limits currently adopted by NASA, and the risk of developing cancer due to radiation exposure after a Mars mission could be two times greater than what scientists previously thought.",
    "Occasional solar proton events (SPEs) produce much higher doses, as observed in September 2017, when NASA reported radiation levels on the surface of Mars were temporarily doubled, and were associated with an aurora 25-times brighter than any observed earlier, due to a massive, and unexpected, solar storm.",
    "Building living quarters underground (possibly in Martian lava tubes) would significantly lower the colonists' exposure to radiation.",
    "Much remains to be learned about space radiation.",
    "In 2003, NASA's Lyndon B. Johnson Space Center opened a facility, the NASA Space Radiation Laboratory, at Brookhaven National Laboratory, that employs particle accelerators to simulate space radiation.",
    "The facility studies its effects on living organisms, as well as experimenting with shielding techniques.",
    "Initially, there was some evidence that this kind of low level, chronic radiation is not quite as dangerous as once thought; and that radiation hormesis occurs.",
    "However, results from a 2006 study indicated that protons from cosmic radiation may cause twice as much serious damage to DNA as previously estimated, exposing astronauts to greater risk of cancer and other diseases.",
    "As a result of the higher radiation in the Martian environment, the summary report of the Review of U.S. Human Space Flight Plans Committee released in 2009 reported that Mars is not an easy place to visit with existing technology and without a substantial investment of resources.",
    "NASA is exploring a variety of alternative techniques and technologies such as deflector shields of plasma to protect astronauts and spacecraft from radiation.",
    "Psychological effects",
    "Due to the communication delays, new protocols need to be developed in order to assess crew members' psychological health.",
    "Researchers have developed a Martian simulation called HI-SEAS (Hawaii Space Exploration Analog and Simulation) that places scientists in a simulated Martian laboratory to study the psychological effects of isolation, repetitive tasks, and living in close-quarters with other scientists for up to a year at a time.",
    "Computer programs are being developed to assist crews with personal and interpersonal issues in absence of direct communication with professionals on Earth.",
    "Current suggestions for Mars exploration and colonization are to select individuals who have passed psychological screenings.",
    "Psychosocial sessions for the return home are also suggested in order to reorient people to society."
]

sentence_embeddings = embedder.embedding_str(sentences)

queries = sentences

queries_embeddings = embedder.embedding_str(queries)

print(cosine_similarity(queries_embeddings, sentence_embeddings))

# Find the closest 10 sentences of the corpus for each query sentence based on cosine similarity
i = 0
for query in queries:
    cos_sim = cosine_similarity(queries_embeddings[i], sentence_embeddings)[0]
    top_res = torch.topk(cos_sim, k=min(len(sentences), 10))
    i = i + 1

    print("Query:", query)
    print("Top 10 most similar sentences:")
    for score, idx in zip(top_res[0], top_res[1]):
        print(sentences[idx], f'(Score: {score:.3f})')
