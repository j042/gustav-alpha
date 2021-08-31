#include <stdio.h>
#include <iostream>
#include "tetgen_wrap.h"

// Object
tetgenio_wrap::tetgenio_wrap(){}


void tetgenio_wrap::LoadArray(int npoints, double* points, int nfaces,
                              int* facearr)
{
  facet *f;
  polygon *p;
  int i, j;
  int count = 0;

  // Allocate memory for points and store them
  numberofpoints = npoints;
  pointlist = new double[npoints*3];

  for(i = 0; i < npoints*3; i++) {
    pointlist[i] = points[i];
  }

  // Store the number of faces and allocate memory
  numberoffacets = nfaces;
  facetlist = new tetgenio::facet[nfaces];

  // Load in faces as facets
  for (i = 0; i < nfaces; i++) {
    // Initialize a face
    f = &facetlist[i];
    init(f);
    
    // Each facet has one polygon, no hole, and each polygon has a three\
    //vertices
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[1];

    p = &f->polygonlist[0];
    init(p);
    p->numberofvertices = 3;
    p->vertexlist = new int[3];
    for (j = 0; j < 3; j++) {
      p->vertexlist[j] = facearr[count];
      count++;
    }
  }
}

void tetgenio_wrap::LoadPoly(int npoints,
                             double* points,
                             int nfaces,
                             int* nfacepointsarr,
                             int* facearr)
{
  facet *f;
  polygon *p;
  int i, j;
  int count = 0;

  // Allocate memory for points and store them
  numberofpoints = npoints;
  pointlist = new double[npoints*3];

  for(i = 0; i < npoints*3; i++) {
    pointlist[i] = points[i];
  }

  // Store the number of faces and allocate memory
  numberoffacets = nfaces;
  facetlist = new tetgenio::facet[nfaces];

  // Load in faces as facets
  for (i = 0; i < nfaces; i++) {
    // Initialize a face
    f = &facetlist[i];
    init(f);
    
    // Each facet has one polygon, no hole, 
    // and each polygon has `n` vertices
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[1];

    p = &f->polygonlist[0];
    init(p);
    p->numberofvertices = nfacepointsarr[i];
    p->vertexlist = new int[nfacepointsarr[i]];
    for (j = 0; j < nfacepointsarr[i]; j++) {
      p->vertexlist[j] = facearr[count];
      count++;
    }
  }
}

void tetgenio_wrap::LoadInput(int npoints,
                              double* points,
                              int nfaces,
                              int* nfacepointsarr,
                              int* facearr,
                              int* markerarr,
                              int nholes,
                              double* holearr,
                              int nregions,
                              double* regionarr,
                              int nhfacepoints,
                              double* hfacepointsarr,
                              int nhfaces,
                              int* nhfacepolygonarr,
                              int* nhfacepolygonpointsarr, // pperhface
                              int* hfacepolygonarr,
                              int* nhfaceholesarr,
                              double* hfaceholes,
                              int* hfacemarkerarr)
{
  facet *f;
  polygon *p;
  int i, j, k;
  int count = 0; // counts facevertex

  // Allocate memory for points and store them
  numberofpoints = npoints + nhfacepoints;
  pointlist = new REAL[npoints*3 + (nhfacepoints*3)];

  for(i = 0; i < npoints*3; i++) {
    pointlist[i] = points[i];
  }

  // Store the number of faces and allocate memory
  numberoffacets = nfaces + nhfaces;
  facetlist = new tetgenio::facet[nfaces + nhfaces];
  // Allocate memory for markers
  facetmarkerlist = new int[nfaces + nhfaces];

  // Load in faces as facets
  for (i = 0; i < nfaces; i++) {
    // Initialize a face
    f = &facetlist[i];
    init(f);
    
    // Each facet has one polygon, no hole, 
    // and each polygon has `n` vertices
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[1];

    p = &f->polygonlist[0];
    init(p);
    p->numberofvertices = nfacepointsarr[i];
    p->vertexlist = new int[nfacepointsarr[i]];
    for (j = 0; j < nfacepointsarr[i]; j++) {
      p->vertexlist[j] = facearr[count];
      count++;

    }
    // Assign marker for each facet.
    facetmarkerlist[i] = markerarr[i];
  }

  // hole list -> volumetric hole (n, 3)
  if (nholes > 0) {
    numberofholes = nholes;
    holelist = new REAL[nholes * 3];
    for ( i = 0; i < nholes; i++) {
      for ( j = 0; j < 3; j++) {
        holelist[i * 3 + j] = holearr[i * 3 + j];
      }
    }
  }

  // Regions (n, 5)
  if (nregions > 0) {
    numberofregions = nregions;
    regionlist = new REAL[nregions * 5];
    for (i = 0; i < nregions; i++) {
      for (j = 0; j < 5; j++) {
        regionlist[i * 5 + j] = regionarr[i * 5 + j];
      }
    }
  }

  // hfaces - facets with holes
  if (nhfaces > 0) {
    // Further assgin points
    j = 0;
    for(i = npoints*3; i < npoints*3 + (nhfacepoints * 3); i++) {
      pointlist[i] = hfacepointsarr[j];

      j++;
    }

    // Load in faces as facets
    k = 0; // hface counter - `i` here is global.
    count = 0; // hfacepolygon points counter
    int hole_count = 0;
    int polygon_counter = 0;
    for (i = nfaces; i < nfaces + nhfaces; i++) {

    
      // Initialize a face
      f = &facetlist[i];
      init(f);
    
      // Each facet has some polygon, some holes, 
      // and each polygon has some vertices
      f->numberofpolygons = nhfacepolygonarr[k];
      f->polygonlist = new tetgenio::polygon[nhfacepolygonarr[k]];

      // Polygons
      for (j = 0; j < f->numberofpolygons; j++) { 
        p = &f->polygonlist[j];
        init(p);
        p->numberofvertices = nhfacepolygonpointsarr[polygon_counter];

        p->vertexlist = new int[nhfacepolygonpointsarr[k]];
        for (int l = 0; l < nhfacepolygonpointsarr[polygon_counter]; l++) {
          p->vertexlist[l] = hfacepolygonarr[count];
          count++;
        }
        polygon_counter++;
      }

      // Facet holes
      f->numberofholes = nhfaceholesarr[k];
      f->holelist = new REAL[nhfaceholesarr[k] * 3];
      int l = 0;
      for (j = 0; j < nhfaceholesarr[k]; j++) {
        for (l = 0; l < 3; l++) {
            f->holelist[j * 3 + l] = hfaceholes[j * 3 + l];
        }
        hole_count++;
      }

      // Assign marker for each facet.
      //
      //
      //
      facetmarkerlist[i] = hfacemarkerarr[k];
      k++;
    }


  }

}
