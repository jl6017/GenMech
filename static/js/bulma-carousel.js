(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else if(typeof exports === 'object')
		exports["bulmaCarousel"] = factory();
	else
		root["bulmaCarousel"] = factory();
})(typeof self !== 'undefined' ? self : this, function() {
return (function(modules) { // webpackBootstrap
	// The module cache
	var installedModules = {};

	// The require function
	function __webpack_require__(moduleId) {

		// Check if module is in cache
		if(installedModules[moduleId]) {
			return installedModules[moduleId].exports;
		}
		// Create a new module (and put it into the cache)
		var module = installedModules[moduleId] = {
			i: moduleId,
			l: false,
			exports: {}
		};

		// Execute the module function
		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);

		// Flag the module as loaded
		module.l = true;

		// Return the exports of the module
		return module.exports;
	}


	// expose the modules object (__webpack_modules__)
	__webpack_require__.m = modules;

	// expose the module cache
	__webpack_require__.c = installedModules;

	// define getter function for harmony exports
	__webpack_require__.d = function(exports, name, getter) {
		if(!__webpack_require__.o(exports, name)) {
			Object.defineProperty(exports, name, {
				configurable: false,
				enumerable: true,
				get: getter
			});
		}
	};

	// getDefaultExport function for compatibility with non-harmony modules
	__webpack_require__.n = function(module) {
		var getter = module && module.__esModule ?
			function getDefault() { return module['default']; } :
			function getModuleExports() { return module; };
		__webpack_require__.d(getter, 'a', getter);
		return getter;
	};

	// Object.prototype.hasOwnProperty.call
	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };

	// __webpack_public_path__
	__webpack_require__.p = "";

	// Load entry module and return exports
	return __webpack_require__(__webpack_require__.s = 5);
})
([
/* 0 */
function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var addClasses = function addClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.add(cls);
	});
};

var removeClasses = function removeClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.remove(cls);
	});
};

var width = function width(element) {
	return element.getBoundingClientRect().width || element.offsetWidth;
};

var height = function height(element) {
	return element.getBoundingClientRect().height || element.offsetHeight;
};

var outerHeight = function outerHeight(element) {
	var withMargin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

	var height = element.offsetHeight;
	if (withMargin) {
		var style = window.getComputedStyle(element);
		height += parseInt(style.marginTop) + parseInt(style.marginBottom);
	}
	return height;
};

var css = function css(element, obj) {
	if (!obj) {
		return window.getComputedStyle(element);
	}
	if (obj && typeof obj === 'object') {
		var style = '';
		Object.keys(obj).forEach(function (key) {
			style += key + ': ' + obj[key] + ';';
		});

		element.style.cssText += style;
	}
};

__webpack_require__.d(__webpack_exports__, "a", function() { return css; });
__webpack_require__.d(__webpack_exports__, "b", function() { return height; });
__webpack_require__.d(__webpack_exports__, "c", function() { return outerHeight; });
__webpack_require__.d(__webpack_exports__, "d", function() { return removeClasses; });
__webpack_require__.d(__webpack_exports__, "e", function() { return width; });

},
/* Additional modules would be here but truncated for size */
/* 5 */
function(module, __webpack_exports__, __webpack_require__) {

"use strict";
Object.defineProperty(__webpack_exports__, "__esModule", { value: true });

// Simplified bulma carousel functionality
var bulmaCarousel = {
    attach: function(selector, options) {
        var elements = typeof selector === 'string' ? 
            document.querySelectorAll(selector) : 
            Array.isArray(selector) ? selector : [selector];
            
        var instances = [];
        
        [].forEach.call(elements, function (element) {
            // Basic carousel implementation
            instances.push({
                element: element,
                options: options || {}
            });
        });
        
        return instances;
    }
};

__webpack_exports__["default"] = bulmaCarousel;

}
])["default"];
});
