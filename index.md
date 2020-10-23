---
layout: home
title: Machine Learning and Adaptive Intelligence
---

Welcome to the course **Machine Learning and Adaptive Intelligence**. This course was originally delivered at the University of Sheffield (2011-2015), but the material has been updated to make it available as a resource for students wanting to satisfy prerequisites for other courses.

## Lectures

{% assign lastone = site.lectures | last %}
{% for lecture in site.lectures %}
{% include listlecture.html %}
{% endfor %}

## Special Topic Lectures

The course is designed to end with special topic lectures that build on the material but be given as guest lectures or advaned topics.

{% assign lastone = site.special_topics | last %}
{% for topic in site.special_topics %}
{% include listtopic.html %}
{% endfor %}
