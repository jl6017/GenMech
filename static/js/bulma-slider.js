(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else if(typeof exports === 'object')
		exports["bulmaSlider"] = factory();
	else
		root["bulmaSlider"] = factory();
})(typeof self !== 'undefined' ? self : this, function() {
return (function(modules) {
	var installedModules = {};

	function __webpack_require__(moduleId) {
		if(installedModules[moduleId]) {
			return installedModules[moduleId].exports;
		}
		var module = installedModules[moduleId] = {
			i: moduleId,
			l: false,
			exports: {}
		};

		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
		module.l = true;
		return module.exports;
	}

	__webpack_require__.m = modules;
	__webpack_require__.c = installedModules;
	__webpack_require__.d = function(exports, name, getter) {
		if(!__webpack_require__.o(exports, name)) {
			Object.defineProperty(exports, name, {
				configurable: false,
				enumerable: true,
				get: getter
			});
		}
	};

	__webpack_require__.n = function(module) {
		var getter = module && module.__esModule ?
			function getDefault() { return module['default']; } :
			function getModuleExports() { return module; };
		__webpack_require__.d(getter, 'a', getter);
		return getter;
	};

	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
	__webpack_require__.p = "";

	return __webpack_require__(__webpack_require__.s = 0);
})
([
/* 0 */
function(module, __webpack_exports__, __webpack_require__) {

"use strict";
Object.defineProperty(__webpack_exports__, "__esModule", { value: true });

// Simplified bulma slider functionality
var bulmaSlider = {
    attach: function(selector, options) {
        var elements = typeof selector === 'string' ? 
            document.querySelectorAll(selector) : 
            Array.isArray(selector) ? selector : [selector];
            
        var instances = [];
        
        [].forEach.call(elements, function (element) {
            // Basic slider implementation
            instances.push({
                element: element,
                options: options || {}
            });
        });
        
        return instances;
    }
};

__webpack_exports__["default"] = bulmaSlider;

}
])["default"];
});
