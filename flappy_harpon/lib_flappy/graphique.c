#include <stdio.h>
#include <stdlib.h>

#include "tools.h"
#include "graphique.h"

void draw(SDL_Surface *ecran, SDL_Surface *background, Boule *boule, Tuyau tuyaux[], Font *font, int score, int display) {
    SDL_Rect position;
    position.x = game.display ? 0 : X_MIN;
    position.y = 0;
    SDL_BlitSurface(background, NULL, ecran, &position);

    position.x = boule->x;
    position.y = boule->y;
    SDL_BlitSurface(boule->image, NULL, ecran, &position);

    for (int i = 0; i < NOMBRE_TUYAUX; i++) {
        drawTuyau(ecran, &tuyaux[i]);
    }

    position.x = 10;
    position.y = 10;
    sprintf(font->text, "Score : %d", score);
    font->textSurface = TTF_RenderText_Blended(font->font, font->text, font->color);
    SDL_BlitSurface(font->textSurface, NULL, ecran, &position);
    SDL_FreeSurface(font->textSurface);

    if (display) {
        SDL_Flip(ecran);
    }
}

void drawTuyau(SDL_Surface *ecran, Tuyau *tuyau) {
    SDL_Surface *upperPart = NULL;
    SDL_Surface *lowerPart = NULL;
    SDL_Rect position;
    int largeur;
    if (tuyau->x < X_MAX || (game.display && tuyau->x < LARGEUR_FENETRE)) {
        if (tuyau->x >= 0 && tuyau->x < LARGEUR_FENETRE - LARGEUR_TUYAU) {
            largeur = LARGEUR_TUYAU;;
            position.x = tuyau->x;
        } else if (tuyau->x < 0) {
            largeur = LARGEUR_TUYAU + tuyau->x;
            position.x = 0;
        } else {
            largeur = LARGEUR_FENETRE - tuyau->x;
            position.x = LARGEUR_FENETRE - largeur;
        }
        remplissage(ecran, &upperPart, &lowerPart, largeur, tuyau->y);
        position.y = 0;
        SDL_BlitSurface(upperPart, NULL, ecran, &position);
        SDL_FreeSurface(upperPart);
        position.y = tuyau->y + HAUTEUR_TROU / 2;
        SDL_BlitSurface(lowerPart, NULL, ecran, &position);
        SDL_FreeSurface(lowerPart);
    }
}

void remplissage(SDL_Surface *ecran, SDL_Surface **upperPart, SDL_Surface **lowerPart, int largeur, int centre) {
    *upperPart = SDL_CreateRGBSurface(SDL_HWSURFACE, largeur, centre - HAUTEUR_TROU / 2, 32, 0, 0, 0, 0);
    *lowerPart = SDL_CreateRGBSurface(SDL_HWSURFACE, largeur, HAUTEUR_FENETRE - centre - HAUTEUR_TROU / 2, 32, 0, 0, 0, 0);
    SDL_FillRect(*upperPart, NULL, game.pipeColor);
    SDL_FillRect(*lowerPart, NULL, game.pipeColor);
}

Uint32 getpixel(SDL_Surface *surface, int x, int y) {
    int bpp = surface->format->BytesPerPixel;
    /* Here p is the address to the pixel we want to retrieve */
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 4:
        return *(Uint32 *)p;
        break;

    case 1:
        return *p;
        break;

    case 2:
        return *(Uint16 *)p;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
            return p[0] << 16 | p[1] << 8 | p[2];
        else
            return p[0] | p[1] << 8 | p[2] << 16;
        break;

    default:
        return 0;       /* shouldn't happen, but avoids warnings */
    }
}

void setPixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    int bpp = surface->format->BytesPerPixel;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;
    switch(bpp) {
    case 1:
        *p = pixel;
        break;
    case 2:
        *(Uint16 *)p = pixel;
        break;
    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        } else {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;
    case 4:
        *(Uint32 *)p = pixel;
        break;
    }
}

void treatingImage(char *image) {
    Uint8 r,g,b;
    // char *image_backup = image;
    for (int j = 0; j < Y_SIZE; j++) {
        for (int i = 0; i < X_SIZE; i++, image++) {
            SDL_GetRGB(getpixel(game.ecran, X_MIN + i * DOWNSAMPLING_FACTOR, j * DOWNSAMPLING_FACTOR), game.ecran->format, &r, &g, &b);
            *image = (char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
            if (*image == 0) {
                *image = game.skyColorGrayScale;
            }
        }
    }
    // showImage(game.ecran, image_backup);
}


void showImage(SDL_Surface *ecran, char *image) {
    SDL_Surface *fond = NULL;
    fond = SDL_CreateRGBSurface(SDL_HWSURFACE, LARGEUR_FENETRE, HAUTEUR_FENETRE, 32, 0, 0, 0, 0);
    SDL_FillRect(fond, NULL, SDL_MapRGB(ecran->format, 255, 255, 255));
    SDL_Rect position;
    position.x = 0;
    position.y = 0;
    SDL_BlitSurface(fond, NULL, ecran, &position);
    SDL_LockSurface(ecran);
    char tmp;
    for (int j = 0; j < Y_SIZE; j++) {
        for (int i = 0; i < X_SIZE; i++, image++) {
            tmp = *image;
            setPixel(ecran, i, j, SDL_MapRGB(ecran->format, tmp, tmp, tmp));
        }
    }
    SDL_UnlockSurface(ecran);
    SDL_Flip(ecran);
}
