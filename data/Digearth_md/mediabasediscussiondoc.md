# Mediabase

Oliver

Hockenhull/Jeff Berryman

This is a discussion document that

offers a new research focus for CISR's (The Centre for

Image and Sound Research) collaborative multimedia

program.

This theme is relevant to the post production editing of

multimedia and traditional video/film, and to more general

questions of multimedia interactivity.  This paper also

intends on explicating some of the formal properties of

the new mediums and suggests the application of this

understanding through novel software design principles.The

basic idea is to apply database technology to the creation

andstructuring of media art works.  The hypotheses are

that media art works can be usefully represented as

databases, that such databases ("mediabases") can be used

as practical tools to support shared invention,

production, and delivery of both interactive and linear

media works, and that database science allows for a new

class of forms for interactive work.

Code,

Structure, Script

A mediabase is a computer database

whose purpose is to store a media work or information

pertaining to the structure and creation of a media work.

The mediabase will give access to the formal

characteristics of a new media work and may blend

production content, script, and, in some cases, production

data.  In an interactive work a viewer's interventions

would be implemented as database retrieval operations, and

the presented images and sounds would be data extracted

from the mediabase.  For a traditional work (film or

video) edited in a non-linear editing suite,  the

mediabase would, allow for editors to automate some of the

task presently processed through the labours of editing

decision lists.A number of current interactive media

works, notably nonfiction works that deal with archives,

are concerned with presenting contents of informational

databases.  Such databases, which we call domain

databases, are not complete mediabases in the sense that

we are using the term here, because they contain the

content of the work (or part of it), but not any script,

production, or presentation data.  Every media work

involves script, production, presentation, and content

data.  In current practice, this data is recorded in

various ad-hoc ways.  Under the mediabase concept, this

data is organised and integrated in machine-readable form,

in a way that allows the use of contemporary database

tools.  The extent to which the data can be captured in

this way and the extent to which it must be transformed to

be integrated effectively are research issues.

Research Opportunities

Taking the database approach to media works might lead us

down a number of research avenues: Mediabases would be good for supporting collaborative

work, in that a mediabase can have specific elements that

are created, maintained, and shared in organized ways by

the various individuals and teams involved with the

creation process. Research in this area would entail

analysis of media works creation processes, and design of

appropriate database structures to support the required

collaborative relationships.  An important challenge would

be to develop common concepts for minimizing the number of

different database designs needed to support the various

media and styles in current practice.Database technology

is presently highly evolved in the areas of data sharing

and distributed access.  This represents a great potential

advantage, one which could help this project produce

usable early results.

[Navigation](syntacticaltools.html)

Mediabases

are not intended to be only administrative tools for

organizing production teams.  The intent is to develop

data design and programming principles that render

mediabase systems capable of implementing aesthetic

principles in media works.  One project in the aesthetic

area would apply database techniques to the problem of

selecting paths through navigable works.  At present, such

paths usually must be completely preprogrammed.  As the

complexity of interactive genres increases, this will

become less and less feasible.  Where discrete works or

fragments are merged, preprogrammed navigation may be

completely impossible without re-doing whole productions.

Also, preprogramming may in some cases  be aesthetically

undesirable.  In general, there is a need for computer

assistance in the making of navigation decisions.If we

think of a linear work as a particular traversal of a

navigable one, it becomes evident that navigating and

editing have much in common.  Montage theory describes the

meanings of cinematic transitions (editing) in terms of

the attributes of the shots and scenes involved.  This

project would seek to apply montage theory to the

navigation problem, using a database implementation

approach.  The research would first devise a methodology

for representing shot and scene attributes in relational

database form, then develop software to use those

attributes for helping make navigation decisions in

interactive works.  This software would present to the

user, at each interactive decision point, those navigation

alternatives that would tend to have the most cinematic

meaning.  Thus, the interactive work would tend to foster

valid viewer experiences, regardless of how much

preprogramming had been done or not done.In the project

plan we wish to develop several stages of montage software

development, beginning with simple algorithmic code, and

ending with mixed-initiative expert systems.This project

concept is based principally on the ideas of Oliver

Hockenhull, a Vancouver filmmaker, writer and communition

theorist, who has for several years been working on the

concept of applying montage theory to interactive works.

-----

The Associative Work &amp; Self-Contextualization

Most of today's interactive media

work is based on the hypermedia concept, in which data

elements can contain one or more pointers that sviewers can

follow to reach other such elements.  Thus, a hypermedia

document is a network, in which data elements form a

multiply connected nonhierarchical graph. Navigation,

the process of following pointers from

[one data element to another](ontotheground.html) ,

is a concept akin to hunting, in which the viewer "goes"

from one place to another, seeking an objective.In order

to produce a hypermedia work capable of delivering a rich

viewer experience, its creators must develop and present a

large and complex context -- a cyberspace...Generally

speaking, creating cyberspaces is time consuming and

expensive.The hypermedia model does not easily support

works based on gathering, rather than hunting, paradigms.

In such works, the counterpart of navigation is

association, in which elements are gathered from global

contexts and arranged to form local contexts. Associative

works are works that offer the viewer the ability to

gather elements from a global context and arrange them

into one or more private contexts.  Association is a

relational activity, in which the viewer builds contexts

through interaction with the work. We call an associative

work self-contextualizing to suggest that each viewer's

particular context is created afresh from the elements of

the work, without starting from a prebuilt global form. We

envision the primary elements of an associative work as

follows:

1.	One or more search spaces, which are

databases from which the viewer may 	extract elements

of content.  The union of all works search spaces is

referred to as its global context.  Many, if not most,

works will have only a single search 	space in their

global contexts.

2.	One or more gathering tools

that the viewer can use to find elements in the global

context.  These tools may be common to all search spaces,

or specific to 	particular ones.

3.	One or

more personal workspaces for each viewer.  For many works,

these workspaces would be non-volatile, to allow

continuity over multiple 	sessions.  The structure of

the workspaces would vary from work to work, and 	could

vary from simple to complex.

4.	One or more

arranging tools that the viewer can use to arrange

gathered 	elements in the workspaces.It is natural to

envision the user interface to such a work as one that

would allow many different objects -- images, animations,

sounds, and so on -- to coexist in a single multimedia

presentation at once.  Which elements were combined and

juxtaposed would vary, depending on the what objects the

viewer had gathered and how she or he had arranged them in

the personal workspace.

Searches, Continuity, and Style

In most associative works, how searches work will

be the strongest single determinant of viewer experience.

We believe that most associative works will support

mixed-initiative searching, i.e. a searching in which

gathering choices are made jointly by the viewer and by

the work itself. One class of mixed-initiative systems

features searches that depend on the contents and

arrangement of the viewer's current personal context.  For

example, the montage approach described above provides a

way for searches to be qualified by cinematic attributes,

these choice of an attribute is selected by the viewer.

Specifically, the searches in a montage-based associative

work are constrained to find only (or mainly) those

elements that are cinematically meaningful in the current

personal context.    In a mixed-initiative associative

system, as a viewer built up her or his own workspace,

successive searches would tend increasingly to find

objects that "worked"in it, contextually.  Thus, the

system could support continuity  and style.  The system

would learn the viewer.

Publication and

Combination

[An attractive](reddust.html)

[outcome](reddust.html)

of the gathering process in associative works

is that the personal workspaces created by the viewers

(or, if you will, co-created by the viewers and the

original authors) are themselves valid artifacts that may

be preserved and shared as searchable contexts.  We say

that a personal workspace offered to others as a

searchable context has been published.Given certain

standards, search spaces that had never "met" each other

before could be combined to create new possibilities.  For

example, CISR could concatenate an associative Cunningham

Archive search space with search spaces from other

artists' archives of the same historical period to allow

viewers to find new associations.Current database

technology is moving toward an object-oriented paradigm,

in which data elements and the programs that use them are

bundled into relatively self-contained units  ("objects").

Because they are self-contained, these units tend to be

more easily shared between contexts, and transported into

new ones.  Thus, having object-oriented mediabases would

facilitate the reuse of published media material, the

combining of disparate media works, and other interesting

integrations.  Making sharing and reuse of work easier has

its risks, in that intellectual property and integrity of

creative results can be more easily compromised.  This

risk would be offset somewhat by the fact that modern

database tools have highly developed access control

facilities which, if properly invoked, can provide highly

selective and secure restrictions.

Project

Scope

This R&amp;D project extends from cinematic theory to

database theory, with the results having immediate

practical application.  The results would depend

critically on effective interweaving of technical and

aesthetic disciplines.  A number of outcomes might be

supported:  Various commercial and/or prototypical

production tools could be developed to serve the different

interests mentioned in Opportunity, above.  Sponsorship of

exploratory and/or demonstration art works and the

development of a montage software package would be a clear

option. To do the subject justice, collaboration with

Banff and other sites inside and outside Rapport would be

indicated.  Finally, the idea would seem to offer good

potential for academic research at the graduate and

possibly faculty level.

Conclusion:

It is fruitful

to consider the digitalization of media creations (film,

video, or interactive multimedia) as a distributed

database problem in which the work is realised as a single

shared database ("Mediabase") that is designed, built, and

used in various ways by the creative team.  For

interactive works, the database and associated programming

codes would also be the work's final form.  For

conventional linear works, the final assembly process

would be a specifically formatted extract of the

database.This approach is distinct from merely using a

multimedia user interface as a way to access an

informational database in which the information in the

database is viewed as domain data.

***Understanding the relationship of aesthetic***

***intent, software structure, and user interaction is a***

***necessary departure for furthering eloquence in the new***

***media.***

[*oliverh@intergate.bc.ca*](mailto:oliverh@intergate.bc.ca)