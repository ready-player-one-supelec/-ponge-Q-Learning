#pragma once

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <SDL/SDL_ttf.h>

#define GRAVITY 0.4
#define LARGEUR_FENETRE 720//1080 // Il faut de préférence une largeur supérieure à 720
#define HAUTEUR_FENETRE 440//720 // Il faut de préférence une hauteur supérieure à 440
#define LARGEUR_TUYAU 100
#define HAUTEUR_TROU 300
#define TROU_CENTRE_YMIN (25+(HAUTEUR_TROU/2))
#define TROU_CENTRE_YMAX (HAUTEUR_FENETRE - (HAUTEUR_TROU/2) - 100)
#define PAS_ENTRE_TUYAU 300
#define TOLERANCE 2 //nombre de pixels de tolérance pour les collisions
#define ESPACE_INTER_TUYAU (PAS_ENTRE_TUYAU+LARGEUR_TUYAU)
#define NOMBRE_TUYAUX (2+(LARGEUR_FENETRE/ESPACE_INTER_TUYAU))
#define BOULE_XAXIS 100

// IMAGE GIVEN TO THE AI
#define DOWNSAMPLING_FACTOR 10
#define X_MIN BOULE_XAXIS
#define X_MAX (BOULE_XAXIS + PAS_ENTRE_TUYAU)
#define X_SIZE ((X_MAX - X_MIN) / DOWNSAMPLING_FACTOR)
#define Y_SIZE (HAUTEUR_FENETRE / DOWNSAMPLING_FACTOR)

#define SKY_RED 135
#define SKY_GREEN 206
#define SKY_BLUE 235

enum {WAIT, JUMP};

typedef struct Boule Boule;
struct Boule {
    SDL_Surface *image;
    float x;
    float vx;
    float y;
    float vy;
    int height;
};

typedef struct Tuyau Tuyau;
struct Tuyau {
    float x;
    float y;
};

typedef struct Font Font;
struct Font {
    SDL_Color color;
    TTF_Font *font;
    SDL_Surface *textSurface;
    char text[31];
};

typedef struct Game Game;
struct Game {
    SDL_Surface *ecran;
    SDL_Surface *background;
    Boule boule;
    Font font;
    int score;
    Tuyau tuyaux[NOMBRE_TUYAUX];
    int display;
    Uint32 pipeColor;
    Uint32 skyColor;
    char skyColorGrayScale;
    int stepSurvived;
};

int min(int v1, int v2);
int max(int v1, int v2);
int randCenter(void);
