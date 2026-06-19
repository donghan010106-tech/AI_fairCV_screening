%% sn-jnl.cls -- OFFLINE-BUILD SHIM for the Springer Nature journal template.
%%
%% This is NOT the official class. It implements the same author-facing macro
%% API (\title, \author*[..]{\fnm..\sur..}, \affil*[..]{\orgdiv..\orgname..},
%% \abstract{}, \keywords{}, \maketitle, \bmhead{}, theorem environments, etc.)
%% on top of `article`, so this paper compiles anywhere offline. For submission,
%% delete this file and use the OFFICIAL `sn-jnl.cls` from
%%   https://www.springernature.com/gp/authors/campaigns/latex-author-support
%% (or the "Springer Nature LaTeX Template" on Overleaf) -- main.tex is written
%% to the official API and needs no changes.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/06/15 v0-shim Springer Nature journal (offline build shim)]
\DeclareOption*{\relax}     % swallow sn-mathphys-num, pdflatex, Numbered, etc.
\ProcessOptions\relax
\LoadClass[10pt,twoside]{article}

\RequirePackage{fontspec}
  \setmainfont{TeX Gyre Termes}
  \setsansfont{TeX Gyre Heros}
  \setmonofont{TeX Gyre Cursor}[Scale=0.92]
\RequirePackage[a4paper,top=2.6cm,bottom=2.8cm,left=2.4cm,right=2.4cm]{geometry}
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{graphicx,booktabs,multirow,array}
\RequirePackage{caption}\captionsetup{font=small,labelfont=bf}
\renewcommand{\figurename}{Fig.}
\RequirePackage{subcaption}
\RequirePackage[table]{xcolor}
\RequirePackage[ruled,vlined,linesnumbered]{algorithm2e}
\RequirePackage{tikz}
\RequirePackage{enumitem}
\RequirePackage{microtype}

\setlength{\parindent}{1.2em}
\renewcommand{\baselinestretch}{1.13}
\tolerance=1200 \emergencystretch=2em

% --- front-matter API ----------------------------------------------------- %
\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\spfx}[1]{#1}\newcommand{\sfx}[1]{#1}
\newcommand{\orgdiv}[1]{#1}\newcommand{\orgname}[1]{#1}
\newcommand{\orgaddress}[1]{#1}\newcommand{\street}[1]{#1}
\newcommand{\city}[1]{#1}\newcommand{\postcode}[1]{#1}
\newcommand{\state}[1]{#1}\newcommand{\country}[1]{#1}
\newcommand{\email}[1]{\gdef\sn@email{#1}}
\newcommand{\sn@email}{}
\newcommand{\jyear}[1]{}
\newcommand{\titlerunning}[1]{}\newcommand{\authorrunning}[1]{}

\newcounter{sn@aff}
\newcommand{\sn@authors}{}
\newcommand{\sn@affils}{}
\newcommand{\sn@corr}{}
% \author*[id]{name}  and  \author[id]{name}
\renewcommand\author{\@ifstar\sn@authorstar\sn@author}
\newcommand{\sn@author}[2][]{%
  \g@addto@macro\sn@authors{#2\textsuperscript{#1}\quad}}
\newcommand{\sn@authorstar}[2][]{%
  \g@addto@macro\sn@authors{#2\textsuperscript{#1}\quad}%
  \gdef\sn@hascorr{1}}
% \affil*[id]{...}  and  \affil[id]{...}
\newcommand{\affil}{\@ifstar\sn@affilstar\sn@affil}
\newcommand{\sn@affil}[2][]{%
  \g@addto@macro\sn@affils{\textsuperscript{#1}#2\par}}
\newcommand{\sn@affilstar}[2][]{%
  \g@addto@macro\sn@affils{\textsuperscript{#1}#2\par}}

\newcommand{\sn@abstract}{}
\renewcommand{\abstract}[1]{\gdef\sn@abstract{#1}}
\newcommand{\sn@keywords}{}
\newcommand{\keywords}[1]{\gdef\sn@keywords{#1}}

\renewcommand{\maketitle}{%
  \thispagestyle{empty}%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vspace{0.8em}
    {\large\sn@authors\par}\vspace{0.5em}
    {\small\sn@affils}\vspace{0.2em}
    \ifx\sn@email\@empty\else{\small Corresponding author: \texttt{\sn@email}\par}\fi
  \end{center}
  \vspace{0.6em}
  \noindent\hrulefill\par\vspace{0.4em}
  \noindent{\small\textbf{Abstract}\quad\sn@abstract\par}%
  \ifx\sn@keywords\@empty\else
    \vspace{0.4em}\noindent{\small\textbf{Keywords:}\ \sn@keywords\par}\fi
  \vspace{0.3em}\noindent\hrulefill\par\vspace{0.9em}
}

% bold inline heading used by sn-jnl in place of \paragraph
\newcommand{\bmhead}[1]{\par\vspace{0.4em}\noindent\textbf{#1}\quad}

% theorem environments (sn-jnl provides these)
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\endinput
