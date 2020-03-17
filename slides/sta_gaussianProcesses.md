{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "# Week 12: Gaussian Processes\n",
    "\n",
    "### Neil D. Lawrence\n",
    "\n",
    "### 15th December 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Replace with material from GPSS (in prep)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pods\n",
    "import matplotlib.pyplot as plt\n",
    "import mlai\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Review\n",
    "\n",
    "* Last week: Logistic Regression and Generalised Linear Models\n",
    "* Introduced link functions and different transformations.\n",
    "* Showed examples in classification and mentioned possibilities for disease rate models (Come to tonight's Data Hide for more!).\n",
    "* This week: \n",
    "    * Gaussian Processes: non parametric Bayesian modelling\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Gaussian Processes\n",
    "\n",
    "* Basis function models give non-linear predictions.\n",
    "* Need to choose number and location of basis functions. \n",
    "* Gaussian processes is a general framework (basis functions special case)\n",
    "* Within the framework you can consider models with infinite basis functions.\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Sampling a Function\n",
    "\n",
    "**Multi-variate Gaussians**\n",
    "\n",
    "* We will consider a Gaussian with a particular structure of covariance\n",
    "    matrix.\n",
    "* Generate a single sample from this 25 dimensional Gaussian distribution, $\\mathbf{f}=\\left[f_{1},f_{2}\\dots f_{25}\\right]$.\n",
    "\n",
    "* We will plot these points against their index.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "# %load -s compute_kernel mlai.py\n",
    "def compute_kernel(X, X2, kernel, **kwargs):\n",
    "    K = np.zeros((X.shape[0], X2.shape[0]))\n",
    "    for i in np.arange(X.shape[0]):\n",
    "        for j in np.arange(X2.shape[0]):\n",
    "            K[i, j] = kernel(X[i, :], X2[j, :], **kwargs)\n",
    "        \n",
    "    return K\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "# %load -s exponentiated_quadratic mlai.py\n",
    "def exponentiated_quadratic(x, x_prime, variance=1., lengthscale=1.):\n",
    "    \"Exponentiated quadratic covariance function.\"\n",
    "    squared_distance = ((x-x_prime)**2).sum()\n",
    "    return variance*np.exp((-0.5*squared_distance)/lengthscale**2)        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=((5,5)))\n",
    "x = np.linspace(-1, 1, 25)[:, None]\n",
    "K = compute_kernel(x, x, exponentiated_quadratic, lengthscale=0.3)\n",
    "ax.matshow(K)\n",
    "ax.set_xlabel('$i$',fontsize=16)\n",
    "ax.set_ylabel('$j$',fontsize=16)\n",
    "#ax.set_co"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "close all\n",
    "K = kernCompute(kern, x);\n",
    "ax = axes('position', [0 0 1 1])\n",
    "plotMatrix(K, ax, 'none', 'imagesc')\n",
    "colorbar\n",
    "\n",
    "t = [];\n",
    "t = [t xlabel('$\\dataIndex$')];\n",
    "t = [t ylabel('$\\dataIndexTwo$')];\n",
    "printLatexPlot('gpSampleCovariance', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "clf\n",
    "% need to take the real part of the sample as the kernel is numerically less than full rank \n",
    "f = real(gsamp(zeros(1, size(x, 1)), K, 1))';\n",
    "\n",
    "a = plot(f, 'x', 'color', blackColor);\n",
    "t = [t xlabel('$\\dataIndex$')];\n",
    "t = [t ylabel('$\\mappingFunction_\\dataIndex$')];\n",
    "set(gca, 'ylim', [-2 2])\n",
    "set(gca, 'xlim', [0 26])\n",
    "set(gca, 'ytick', [-2 -1 0 1 2])\n",
    "set(a,'markersize', 5)\n",
    "set(a, 'linewidth', 6)\n",
    "box off\n",
    "xlim = get(gca, 'xlim');\n",
    "ylim = get(gca, 'ylim');\n",
    "line([xlim(1) xlim(1)], ylim, 'color', blackColor)\n",
    "line(xlim, [ylim(1) ylim(1)], 'color', blackColor)\n",
    "\n",
    "printLatexPlot('gpSampleValue', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "yLim = get(gca, 'ylim');\n",
    "xLim = get(gca, 'xlim');\n",
    "ySpan = yLim(2) - yLim(1);\n",
    "xSpan = xLim(2) - xLim(1);\n",
    "chand = oval([1 f(1)], 0.025*xSpan, 0.025*ySpan);\n",
    "chand = [chand oval([2 f(2)], 0.025*xSpan, 0.025*ySpan)];\n",
    "set(chand, 'color', magentaColor, 'linewidth', 3)\n",
    "printLatexPlot('gpSampleValue12', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "clf\n",
    "options = plotMatrixOptions;\n",
    "options.color = blackColor;\n",
    "options.highlight.on = true;\n",
    "options.highlight.row = [1 2];\n",
    "options.highlight.col= [1 2];\n",
    "options.highlight.color = magentaColor;\n",
    "options.highlight.width = 10;\n",
    "ax = axes('position', [0 0 1 1])      \n",
    "plotMatrix(K, ax, 'none', 'imagesc', options)\n",
    "colorbar\n",
    "printLatexPlot('gpSampleCovariance12a', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "options.zoom.on = true;\n",
    "options.zoom.row = [1 10];\n",
    "options.zoom.col = [1 10];\n",
    "\n",
    "cla(ax)\n",
    "plotMatrix(K, ax, 'none', 'imagesc', options)\n",
    "colorbar\n",
    "printLatexPlot('gpSampleCovariance12b', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "options.zoom.on = true;\n",
    "options.zoom.row = [1 5];\n",
    "options.zoom.col = [1 5];\n",
    "\n",
    "cla(ax)\n",
    "plotMatrix(K, ax, 'none', 'imagesc', options)\n",
    "colorbar\n",
    "printLatexPlot('gpSampleCovariance12c', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "options.zoom.on = true;\n",
    "options.zoom.row = [1 3];\n",
    "options.zoom.col = [1 3];\n",
    "\n",
    "plotMatrix(K, ax, 'none', 'imagesc', options)\n",
    "colorbar \n",
    "printLatexPlot('gpSampleCovariance12d', '../../../gp/tex/diagrams/', 0.4*\n",
    "textWidth);\n",
    "\n",
    "options.zoom.on = true;\n",
    "options.zoom.row = [1 2];\n",
    "options.zoom.col = [1 2];\n",
    "\n",
    "plotMatrix(K, ax, 'none', 'imagesc', options)\n",
    "colorbar \n",
    "printLatexPlot('gpSampleCovariance12e', '../../../gp/tex/diagrams/', 0.4*\n",
    "textWidth);\n",
    "\n",
    "clf\n",
    "options.highlight.on = false;\n",
    "options.zoom.on = false;\n",
    "options.bracketWidth = 10;\n",
    "ax = axes('position', [0 0 1 1])      \n",
    "plotMatrix(K([1 2], [1 2]), ax, 'square', 'values', options)\n",
    "printLatexPlot('gpSampleCovariance12f', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "\n",
    "clf\n",
    "options.highlight.on = false;\n",
    "options.zoom.on = false;\n",
    "options.bracketWidth = 10;\n",
    "ax = axes('position', [0 0 1 1])      \n",
    "plotMatrix(K([1 5], [1 5]), ax, 'square', 'values', options)\n",
    "printLatexPlot('gpSampleCovariance15f', '../../../gp/tex/diagrams/', 0.4*textWidth);\n",
    "\n",
    "save gpdistfunc K x f\n",
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Gaussian Distribution Sample\n",
    "  \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "## Log Odds\n",
    "\n",
    "* model the *log-odds* with the basis functions.\n",
    "* [odds](http://en.wikipedia.org/wiki/Odds) are defined as the ratio of the probability of a positive outcome, to the probability of a negative outcome. \n",
    "* Probability is between zero and one, odds are:\n",
    "    $$ \\frac{\\pi}{1-\\pi} $$\n",
    "* Odds are between $0$ and $\\infty$. \n",
    "* Logarithm of odds maps them to $-\\infty$ to $\\infty$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Logit Link Function\n",
    "\n",
    "* The [Logit function](http://en.wikipedia.org/wiki/Logit), $$g^{-1}(\\pi_i) = \\log\\frac{\\pi_i}{1-\\pi_i}.$$ This function is known as a *link function*.\n",
    "\n",
    "* For a standard regression we take,\n",
    "    $$f(\\mathbf{x}_i) = \\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i),$$\n",
    "* For classification we perform a logistic regression. \n",
    "    $$\\log \\frac{\\pi_i}{1-\\pi_i} = \\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i)$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Inverse Link Function\n",
    "\n",
    "We have defined the link function as taking the form $g^{-1}(\\cdot)$ implying that the inverse link function is given by $g(\\cdot)$. Since we have defined,\n",
    "$$\n",
    "g^{-1}(\\pi(\\mathbf{x})) = \\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\n",
    "$$\n",
    "we can write $\\pi$ in terms of the *inverse link* function, $g(\\cdot)$ as \n",
    "$$\n",
    "\\pi(\\mathbf{x}) = g(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})).\n",
    "$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(14,7))\n",
    "f = np.linspace(-8, 8, 100)\n",
    "g = 1/(1+np.exp(-f))\n",
    "\n",
    "ax.plot(f, g, 'r-')\n",
    "ax.set_title('Logistic Function', fontsize=20)\n",
    "ax.set_xlabel('$f_i$', fontsize=20)\n",
    "ax.set_ylabel('$g_i$', fontsize=20)\n",
    "plt.savefig('./diagrams/logistic.svg')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Logistic function\n",
    "\n",
    "* [Logistic](http://en.wikipedia.org/wiki/Logistic_function) (or sigmoid) squashes real line to between 0   & 1. Sometimes also called a 'squashing function'. \n",
    "![](./diagrams/logistic.svg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "\n",
    "### Prediction Function\n",
    "* Can now write $\\pi$ as a function of the input and the parameter vector as, $$\\pi(\\mathbf{x},\\mathbf{w}) = \\frac{1}{1+ \\exp\\left(-\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\\right)}.$$\n",
    "\n",
    "* Compute the output of a standard linear basis function composition ($\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})$, as we did for linear regression)\n",
    "\n",
    "* Apply the inverse link function, $g(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}))$. \n",
    "\n",
    "* Use this value in a Bernoulli distribution to form the likelihood. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Bernoulli Reminder\n",
    "\n",
    "* From last time $$P(y_i|\\mathbf{w}, \\mathbf{x}) = \\pi_i^{y_i} (1-\\pi_i)^{1-y_i}$$\n",
    "\n",
    "* Trick for switching betwen probabilities\n",
    "```python\n",
    "def bernoulli(y, pi):\n",
    "    if y == 1:\n",
    "        return pi\n",
    "    else:\n",
    "        return 1-pi\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Maximum Likelihood\n",
    "\n",
    "\n",
    "* Conditional independence of data:$$P(\\mathbf{y}|\\mathbf{w}, \\mathbf{X}) = \\prod_{i=1}^n P(y_i|\\mathbf{w}, \\mathbf{x}_i). $$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Log Likelihood\n",
    "\n",
    "\\begin{align*}\n",
    "\\log P(\\mathbf{y}|\\mathbf{w}, \\mathbf{X}) = & \\sum_{i=1}^n \\log P(y_i|\\mathbf{w}, \\mathbf{x}_i) \\\\ = &\\sum_{i=1}^n y_i \\log \\pi_i \\\\ & + \\sum_{i=1}^n (1-y_i)\\log (1-\\pi_i)\n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Objective Function\n",
    "\n",
    "* Probability of positive outcome for the $i$th data point $$\\pi_i = g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i)\\right),$$ where $g(\\cdot)$ is the *inverse* link function\n",
    "\n",
    "* Objective function of the form \\begin{align*}\n",
    "E(\\mathbf{w}) = & -  \\sum_{i=1}^n y_i \\log g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i)\\right) \\\\& - \\sum_{i=1}^n(1-y_i)\\log \\left(1-g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i)\\right)\\right).\n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Minimize Objective\n",
    "\n",
    "* Grdient wrt  $\\pi(\\mathbf{x};\\mathbf{w})$ \n",
    "\\begin{align*}\n",
    "\\frac{\\text{d}E(\\mathbf{w})}{\\text{d}\\mathbf{w}} = & -\\sum_{i=1}^n \\frac{y_i}{g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\\right)}\\frac{\\text{d}g(f_i)}{\\text{d}f_i} \\boldsymbol{\\phi(\\mathbf{x}_i)} \\\\ & +  \\sum_{i=1}^n \\frac{1-y_i}{1-g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\\right)}\\frac{\\text{d}g(f_i)}{\\text{d}f_i} \\boldsymbol{\\phi(\\mathbf{x}_i)}\n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Link Function Gradient\n",
    "\n",
    "* Also need gradient of inverse link function wrt parameters.\n",
    "\\begin{align*}\n",
    "g(f_i) &= \\frac{1}{1+\\exp(-f_i)}\\\\\n",
    "&=(1+\\exp(-f_i))^{-1}\n",
    "\\end{align*}\n",
    "and the gradient can be computed as\n",
    "\\begin{align*}\n",
    "\\frac{\\text{d}g(f_i)}{\\text{d} f_i} & = \\exp(-f_i)(1+\\exp(-f_i))^{-2}\\\\\n",
    "& = \\frac{1}{1+\\exp(-f_i)} \\frac{\\exp(-f_i)}{1+\\exp(-f_i)} \\\\\n",
    "& = g(f_i) (1-g(f_i))\n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Objective Gradient\n",
    "\n",
    "\\begin{align*}\n",
    "\\frac{\\text{d}E(\\mathbf{w})}{\\text{d}\\mathbf{w}} = & -\\sum_{i=1}^n y_i\\left(1-g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\\right)\\right) \\boldsymbol{\\phi(\\mathbf{x}_i)} \\\\ & + \\sum_{i=1}^n (1-y_i)\\left(g\\left(\\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x})\\right)\\right) \\boldsymbol{\\phi(\\mathbf{x}_i)}.\n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Optimization of the Function\n",
    "\n",
    "* Can't find a stationary point of the objective function analytically.\n",
    "\n",
    "* Optimization has to proceed by *numerical methods*. \n",
    "    * [Newton's method](http://en.wikipedia.org/wiki/Newton%27s_method) or \n",
    "    * [gradient based optimization methods](http://en.wikipedia.org/wiki/Gradient_method) \n",
    "    \n",
    "* Similarly to matrix factorization, for large data *stochastic gradient descent*  (Robbins Munroe optimization procedure) works well. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Ad Matching for Facebook\n",
    "\n",
    "* This approach used in many internet companies.\n",
    "\n",
    "* Example: ad matching for Facebook.\n",
    "    * Millions of advertisers\n",
    "    * Billions of users\n",
    "    * How do you choose who to show what?\n",
    "    \n",
    "* Logistic regression used in combination with [decision trees]()\n",
    "\n",
    "* [Paper available here](http://www.herbrich.me/papers/adclicksfacebook.pdf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Other GLMs\n",
    "\n",
    "* Logistic regression is part of a family known as *generalized linear models*\n",
    "\n",
    "* They all take the form $$g^{-1}(f_i(x)) = \\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}_i)$$\n",
    "\n",
    "* As another example let's look at *Poisson regression*.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Poisson Distribution\n",
    "\n",
    "* Poisson distribution is used for 'count data'. For non-negative integers, $y$, $$P(y) = \\frac{\\lambda^y}{y!}\\exp(-y)$$\n",
    "\n",
    "* Here $\\lambda$ is a *rate* parameter that can be thought of as the number of arrivals per unit time.\n",
    "\n",
    "* Poisson distributions can be used for disease count data. E.g. number of incidence of malaria in a district.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "slideshow": {
     "slide_type": "skip"
    }
   },
   "outputs": [],
   "source": [
    "from scipy.stats import poisson\n",
    "fig, ax = plt.subplots(figsize=(14,7))\n",
    "y = np.asarray(range(0, 16))\n",
    "p1 = poisson.pmf(y, mu=1.)\n",
    "p3 = poisson.pmf(y, mu=3.)\n",
    "p10 = poisson.pmf(y, mu=10.)\n",
    "\n",
    "ax.plot(y, p1, 'r.-', markersize=20, label='$\\lambda=1$')\n",
    "ax.plot(y, p3, 'g.-', markersize=20, label='$\\lambda=3$')\n",
    "ax.plot(y, p10, 'b.-', markersize=20, label='$\\lambda=10$')\n",
    "ax.set_title('Poisson Distribution', fontsize=20)\n",
    "ax.set_xlabel('$y_i$', fontsize=20)\n",
    "ax.set_ylabel('$p(y_i)$', fontsize=20)\n",
    "ax.legend(fontsize=20)\n",
    "plt.savefig('./diagrams/poisson.svg')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Poisson Distribution\n",
    "\n",
    "![](./diagrams/poisson.svg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Poisson Regression\n",
    "\n",
    "* In a Poisson regression make rate a function of space/time.$$\\log \\lambda(\\mathbf{x}, t) = \\mathbf{w}_x^\\top \\boldsymbol{\\phi_x(\\mathbf{x})} + \\mathbf{w}_t^\\top \\boldsymbol(\\phi_t(t))$$\n",
    "\n",
    "* This is known as a *log linear* or *log additive* model. \n",
    "\n",
    "* The link function is a logarithm.\n",
    "\n",
    "* We can rewrite such a function as \n",
    "$$\\log \\lambda(\\mathbf{x}, t) = f_x(\\mathbf{x}) + f_t(t)$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "### Multiplicative Model\n",
    "\n",
    "* Be careful though ... a log additive model is really multiplicative.\n",
    "$$\\log \\lambda(\\mathbf{x}, t) = f_x(\\mathbf{x}) + f_t(t)$$\n",
    "\n",
    "* Becomes $$\\lambda(\\mathbf{x}, t) = \\exp(f_x(\\mathbf{x}) + f_t(t))$$\n",
    "\n",
    "* Which is equivalent to  $$\\lambda(\\mathbf{x}, t) = \\exp(f_x(\\mathbf{x}))\\exp(f_t(t))$$\n",
    "\n",
    "* Link functions can be deceptive in this way."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "#### Reading\n",
    "\n",
    "- Section 5.2.2 of @Rogers:book11 up to pg 182.\n"
   ]
  }
 ],
 "metadata": {
  "celltoolbar": "Slideshow",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
