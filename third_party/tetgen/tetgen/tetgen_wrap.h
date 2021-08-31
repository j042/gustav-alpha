#include "tetgen.h"


// Adds additional functionality to original tetgen object
class tetgenio_wrap : public tetgenio
{
    public:
        //constructor
        tetgenio_wrap();

        facet *f;
        polygon *p;
//        int markers;
        
        void LoadArray(int, double*, int, int*);
        void LoadPoly(int npoints,
                      double* points, 
                      int nfaces,
                      int* nfacepointsarr,
                      int* facearr);
        void LoadInput(int npoints, 
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
                       int* hfacemarkerarr);


        //destructor
//        ~myRectangle();

};

