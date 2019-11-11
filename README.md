# Image Analysis San Salvador
By Fabiola Alba Vivar (Brown U) and Furkan Top (MIT)

We use Satellite Data to study Gang Presence in San Salvador. We obtained Google Earth Satellite Images from houses and neighborhoods inside and outside gang dominated areas of the city. We want to revise whether there are systematic differences between these areas using image analysis. We also use Machine Learning tools in order to predict if a neighborhood is gang dominated or not. 
 
## Context 
 
We study the city of San Salvador, El Salvador's capital. This city is located in the Boqueron Volcano Valley. Due to his very tropical climate and hilly topography, it is surrounded by green mountains and rural open space. Figure 1 shows a satellite image of the city. It is estimated to have almost 2 million habitats, increasing at a rate of 1.9 average percent per year since 1999. For more information, visit: http://www.atlasofurbanexpansion.org/cities/view/San\_Salvador

![SanSalvador](SanSalvadorSatViewTest.png)

During the recent decades, the city have suffered the violence of two major gangs: MS-13 and 18th Street. Several parts of the city were completely dominated by these gangs, generating negative outcomes to its habitats. We used the data collected by Melnikov, Schmidt-Padilla and Sviatschi (2019), where they drawn the gang boundaries across the city. Aditionally, they surveyed over 2000 households in gang and non gang areas.The households were selected by dividing the census segments into 30 meter bins, denoting distance to gang territory. The distribution of the households are plotted in Figure 2.

![Fieldwork](Fieldwork_SanSalvador.jpg)

We collected Google Earth Satellite images from each of the households included in the survey sample. Images collected are centered at the GPS location of each household and they were stored as 480x480 RGB files. In total, we collected 2314 images, where 54 percent of them belong to gang dominated area.

We processed the images in Python, and for each one we obtained a 480x480 3D matrix. Then, we re-scaled this data by transforming it into a grey scale and therefore we reduced it to a one-dimensional matrix. Figure 2 shows an example for a Gang and No Gang location and their respective collected data is plotted in a histogram.  
