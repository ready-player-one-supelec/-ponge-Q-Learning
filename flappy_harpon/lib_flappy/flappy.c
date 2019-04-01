#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#define _LIB_FLAPPY
#include "tools.h"
#include "flappy.h"
#include "game.h"

char* init_flappy(int display) {
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    initFont(&game.font);
    game.ecran = SDL_SetVideoMode(LARGEUR_FENETRE, HAUTEUR_FENETRE, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
    SDL_WM_SetCaption("Flappy Bird", NULL);

    game.display = display;
    game.pipeColor = SDL_MapRGB(game.ecran->format, 120, 255, 120);
    game.skyColor = SDL_MapRGB(game.ecran->format, SKY_RED, SKY_GREEN, SKY_BLUE);
    game.skyColorGrayScale = (char)(0.2126 * SKY_RED + 0.7152 * SKY_GREEN + 0.0722 * SKY_BLUE);
    if (game.display) {
        game.background = SDL_CreateRGBSurface(SDL_HWSURFACE, game.ecran->w, game.ecran->h, 32, 0, 0, 0, 0);
    } else {
        game.background = SDL_CreateRGBSurface(SDL_HWSURFACE, (X_MAX - X_MIN), game.ecran->h, 32, 0, 0, 0, 0);
    }
    SDL_FillRect(game.background, NULL, game.skyColor);
    game.boule.image = IMG_Load("Images/boule.png");
    game.boule.height = game.boule.image->w;
    reset_flappy();

    char *pointeur = NULL;
    pointeur = (char *)malloc((X_SIZE * Y_SIZE + 1)* sizeof(char));
    return pointeur;
}

void initBoule(Boule *boule) {
    boule->y = 170;
    boule->vy = 0;
    boule->x = BOULE_XAXIS;
    boule->vx = 3;
}

void reset_flappy(void) {
    initBoule(&game.boule);
    game.tuyaux[0].x = 700 ;
    game.tuyaux[0].y = randCenter();
    game.score = 0;
    game.stepSurvived = 0;

    for (int i = 1; i < NOMBRE_TUYAUX ; i++) {
        nextTuyau(&game.tuyaux[i], &game.tuyaux[(i-1) % NOMBRE_TUYAUX]);
    }
}

void changeDisplay(int display) {
    game.display = display;
}

void exit_flappy(void) {
    SDL_FreeSurface(game.background);
    SDL_FreeSurface(game.boule.image);
    TTF_CloseFont(game.font.font);
    TTF_Quit();
}

void initFont(Font *font) {
    font->font = NULL;
    font->font = TTF_OpenFont("Sugar Addiction - TTF.ttf", 30);
    font->color.r = 255;
    font->color.g = 0;
    font->color.b = 0;
    font->textSurface = NULL;
}

void getSize(int *x_size, int *y_size) {
    *x_size = X_SIZE;
    *y_size = Y_SIZE;
}

void updateFeatures(int *xToPipe, float *yToUpperPipe, float *yToLowerPipe, float *vy, float *yToTop, float *yToBottom) {
    Tuyau nextPipe = game.tuyaux[0];
    for (int i = 1; i < NOMBRE_TUYAUX; i++) {
        if (game.boule.x >= game.tuyaux[i].x - PAS_ENTRE_TUYAU + LARGEUR_TUYAU && game.boule.x < game.tuyaux[i].x + LARGEUR_TUYAU) {
            nextPipe = game.tuyaux[i];
            break;
        }
    }
    *xToPipe = max(0, (int) (nextPipe.x - game.boule.x - game.boule.height));
    *yToUpperPipe = game.boule.y + HAUTEUR_TROU / 2 - nextPipe.y;
    *yToLowerPipe = nextPipe.y + HAUTEUR_TROU / 2 - game.boule.y - game.boule.height;
    *vy = game.boule.vy;
    *yToTop = game.boule.y;
    *yToBottom = HAUTEUR_FENETRE - game.boule.y - game.boule.height;
}
