from pprint import pprint
from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
import pytest
import unittest


class PieceType(Enum):
    KING = 1000
    QUEEN = 9
    ROOK = 5
    BISHOP = 3
    KNIGHT = 3.5
    PAWN = 1


class Color(Enum):
    BLACK = 0
    WHITE = 1


ChessPiece = namedtuple("ChessPiece", ["piece_type", "color"])
BK = ChessPiece(piece_type=PieceType.KING, color=Color.BLACK)
BQ = ChessPiece(piece_type=PieceType.QUEEN, color=Color.BLACK)
BR = ChessPiece(piece_type=PieceType.ROOK, color=Color.BLACK)
BB = ChessPiece(piece_type=PieceType.BISHOP, color=Color.BLACK)
BN = ChessPiece(piece_type=PieceType.KNIGHT, color=Color.BLACK)
BP = ChessPiece(piece_type=PieceType.PAWN, color=Color.BLACK)
WK = ChessPiece(piece_type=PieceType.KING, color=Color.WHITE)
WQ = ChessPiece(piece_type=PieceType.QUEEN, color=Color.WHITE)
WR = ChessPiece(piece_type=PieceType.ROOK, color=Color.WHITE)
WB = ChessPiece(piece_type=PieceType.BISHOP, color=Color.WHITE)
WN = ChessPiece(piece_type=PieceType.KNIGHT, color=Color.WHITE)
WP = ChessPiece(piece_type=PieceType.PAWN, color=Color.WHITE)


class ChessBoard(object):
    def __init__(self):
        # set up the board
        self.board = [[WR, WN, WB, WQ, WK, WB, WN, WR],
                      [WP, WP, WP, WP, WP, WP, WP, WP],
                      [None, None, None, None, None, None, None, None],
                      [None, None, None, None, None, None, None, None],
                      [None, None, None, None, None, None, None, None],
                      [None, None, None, None, None, None, None, None],
                      [BP, BP, BP, BP, BP, BP, BP, BP],
                      [BR, BN, BB, BQ, BK, BB, BN, BR]]
        # initialize the scores of both sides
        self.white_score = 1052
        self.black_score = 1052

    def clear_board(self) -> None:
        self.board = [[None] * 8 for i in range(8)]

    def reset_board(self) -> None:
        self.__init__()

    # for a Black Pawn - can move 1 square vertically, or diagonally to get a piece out, and if it is its first turn, 2 squares
    def move_BP(self, start_col, start_row):
        # make an empty list of all of the possible moves
        moves = []

        # set up initial to indicate if it is the pawn's first move or not
        initial = False

        # if it is in row 6, the pawn has not yet moved and this is its first move
        if start_row == 6:
            initial = True

        # to move diagonally
        # pawn can either move up and to the left (-1, -1) or up and to the right (1,-1)
        diagonals = [[1, -1], [-1, -1]]
        # x and y take on the values in the x and y coordinates
        for x, y in diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # check if there is a piece there to get out
                if self.board[r][c]:
                    if self.board[r][c].color == Color.WHITE:
                        moves.append((r, c))

        # checking if it can move forward one spot
        r = start_row - 1
        # if there is no other piece in that spot
        if not self.board[r][start_col]:
            moves.append((r, start_col))

        # if this is the pawn's first move, it can move 2 spaces forward
        if initial:
            # check that nothing is in that spot or the spot before it (bc it can't jump over a piece)
            if not self.board[r][start_col] and not self.board[r - 1][start_col]:
                moves.append((r - 1, start_col))

        return moves

    # for a White Pawn - can move 1 square vertically, or diagonally to get a piece out, and if it is its first turn, 2 squares
    def move_WP(self, start_col, start_row):
        moves = []
        initial = False
        if start_row == 1:
            initial = True

        # # to move diagonally
        # it can either go diagonally down and to the left (-1, 1) or down and to the right (1,1)
        diagonals = [[1, 1], [-1, 1]]
        for x, y in diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is a piece there to get out
                if self.board[r][c]:
                    if self.board[r][c].color == Color.BLACK:
                        moves.append((r, c))

        # now checking for going straight
        r = start_row + 1
        # if there is no other piece in that spot
        if not self.board[r][start_col]:
            moves.append((r, start_col))

        # if this is the pawn's first move, it can move two spaces
        if initial:
            # check that nothing is in that spot or the spot before it (bc it can't jump over a piece)
            if not self.board[r][start_col] and not self.board[r + 1][start_col]:
                moves.append((r + 1, start_col))

        return moves

    # for a Black Rook - can move horizontally or vertically
    def move_BR(self, start_col, start_row):
        moves = []

        # now checking for going down the board
        for i in range(start_row + 1, 8):
            # if there is no other piece in that spot
            if not self.board[i][start_col]:
                moves.append((i, start_col))
            # if we encounter a white piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[i][start_col].color == Color.WHITE:
                moves.append((i, start_col))
                break
            # there was a black piece and we cannot go further
            else:
                break

        # now for going up the board
        for i in range(start_row - 1, -1, -1):
            # if there is no other piece in that spot
            if not self.board[i][start_col]:
                moves.append((i, start_col))
            # if we encounter a white piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[i][start_col].color == Color.WHITE:
                moves.append((i, start_col))
                break
            # there was a black piece and we cannot go further
            else:
                break

        # move right
        for i in range(start_col, 8):
            # if there is no other piece in that spot
            if not self.board[start_row][i]:
                moves.append((start_row, i))
            # if we encounter a white piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[start_row][i].color == Color.WHITE:
                moves.append((start_row, i))
                break
            # there was a black piece and we cannot go further
            else:
                break

        # move left
        for i in range(start_col - 1, -1, -1):
            # if there is no other piece in that spot
            if not self.board[start_row][i]:
                moves.append((start_row, i))
            # if we encounter a white piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[start_row][i].color == Color.WHITE:
                moves.append((start_row, i))
                break
            # there was a black piece and we cannot go further
            else:
                break
        return moves

    # for a White Rook - can move horizontally or vertically
    def move_WR(self, start_col, start_row) -> list:
        moves = []

        # now checking for going forward down the board
        for i in range(start_row + 1, 8):
            # if there is no other piece in that spot
            if not self.board[i][start_col]:
                moves.append((i, start_col))
            # if we encounter a black piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[i][start_col].color == Color.BLACK:
                moves.append((i, start_col))
                break
            # there was a white piece and we cannot go further
            else:
                break

        # now for going up the board
        for i in range(start_row - 1, -1, -1):
            # if there is no other piece in that spot
            if not self.board[i][start_col]:
                moves.append((i, start_col))
            # if we encounter a black piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[i][start_col].color == Color.BLACK:
                moves.append((i, start_col))
                break
            # there was a white piece and we cannot go further
            else:
                break

        # move right
        for i in range(start_col, 8):
            # if there is no other piece in that spot
            if not self.board[start_row][i]:
                moves.append((start_row, i))
            # if we encounter a black piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[start_row][i].color == Color.BLACK:
                moves.append((start_row, i))
                break
            # there was a white piece and we cannot go further
            else:
                break

        # move left
        for i in range(start_col - 1, -1, -1):
            # if there is no other piece in that spot
            if not self.board[start_row][i]:
                moves.append((start_row, i))
            # if we encounter a black piece we can go there, but we cannot go further (bc it can't jump)
            elif self.board[start_row][i].color == Color.BLACK:
                moves.append((start_row, i))
                break
            # there was a white piece and we cannot go further
            else:
                break
        return moves

    # for a Black Bishop - can move in any direction diagonally
    def move_BB(self, start_col, start_row) -> list:
        moves = []

        # to continue down the board (higher rows) and to the right
        down_right_diagonals = [[i, i] for i in range(1, 8)]
        for x, y in down_right_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
                    break
                # there was a black piece and we cannot go further
                else:
                    break

        # to continue down the board (higher rows) and to the left
        down_left_diagonals = [[-i, i] for i in range(1, 8)]
        for x, y in down_left_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
                    break
                # there was a black piece and we cannot go further
                else:
                    break

        # to continue up the board (lower rows) and to the right
        up_right_diagonals = [[i, -i] for i in range(1, 8)]
        for x, y in up_right_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
                    break
                # there was a black piece and we cannot go further
                else:
                    break

        # to continue up the board (lower rows) and to the left
        up_left_diagonals = [[-i, -i] for i in range(1, 8)]
        for x, y in up_left_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
                    break
                # there was a black piece and we cannot go further
                else:
                    break
        return moves

    # for a White Bishop - can move in any direction diagonally
    def move_WB(self, start_col, start_row) -> list:
        moves = []

        down_right_diagonals = [[i, i] for i in range(1, 8)]
        for x, y in down_right_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
                    break
                # there was a white piece and we cannot go further
                else:
                    break

        down_left_diagonals = [[-i, i] for i in range(1, 8)]
        for x, y in down_left_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
                    break
                # there was a white piece and we cannot go further
                else:
                    break

        up_right_diagonals = [[i, -i] for i in range(1, 8)]
        for x, y in up_right_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
                    break
                # there was a white piece and we cannot go further
                else:
                    break

        up_left_diagonals = [[-i, -i] for i in range(1, 8)]
        for x, y in up_left_diagonals:
            r, c = start_row + y, start_col + x
            # make sure it's in range of the board
            if r in range(8) and c in range(8):
                # if there is no other piece in that spot
                if not self.board[r][c]:
                    moves.append((r, c))
                # if we encounter opposite color we can go there, but we cannot go further (bc it can't jump)
                elif self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
                    break
                # there was a white piece and we cannot go further
                else:
                    break
        return moves

    # for a Black Queen - can move horizontally, vertically, or diagonally
    def move_BQ(self, start_col, start_row) -> list:
        # since the queen can move diagonal in any direction
        # it can do any move the bishop can
        moves = self.move_BB(start_col, start_row)

        # it can also move horizontally or vertically, so in any spot a rook can move
        moves.append(self.move_BR(start_col, start_row))
        return moves

    # for a White Queen - can move horizontally, vertically, or diagonally
    def move_WQ(self, start_col, start_row) -> list:
        # since the queen can move diagonal in any direction
        # it can do any move the bishop can
        moves = self.move_WB(start_col, start_row)

        # it can also move horizontally or vertically, in any spot a rook can move
        moves.append(self.move_WR(start_col, start_row))
        return moves

    # for a Black Knight - in L shape - 2 horizontally and 1 vertically or 1 horizontally and 2 vertically
    def move_BN(self, start_col, start_row) -> list:
        moves = []
        potential_moves = [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2]]
        for x, y in potential_moves:
            r, c = start_row + y, start_col + x
            # check that they are in the range of the board
            if r in range(8) and c in range(8):
                # if there is no piece in that spot or the piece in the spot is white, it can move there
                if not self.board[r][c] or self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
        return moves

    # for a White Knight - in L shape - 2 horizontally and 1 vertically or 1 horizontally and 2 vertically
    def move_WN(self, start_col, start_row) -> list:
        moves = []
        potential_moves = [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2]]
        for x, y in potential_moves:
            r, c = start_row + y, start_col + x
            # check that they are in the range of the board
            if r in range(8) and c in range(8):
                # if there is no piece in that spot or the piece in the spot is black, it can move there
                if not self.board[r][c] or self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
        return moves

    # for a Black King - 1 square in any direction
    def move_BK(self, start_col, start_row) -> list:
        # not accounting for putting the king in check or check-mate
        moves = []
        potential_moves = [[1, 0], [-1, 0], [0, 1], [1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for x, y in potential_moves:
            r, c = start_row + y, start_col + x
            # check that they are in the range of the board
            if r in range(8) and c in range(8):
                # if there is no piece in that spot or the piece in the spot is white, it can move there
                if not self.board[r][c] or self.board[r][c].color == Color.WHITE:
                    moves.append((r, c))
        return moves

    # for a White King - 1 square in any direction
    def move_WK(self, start_col, start_row) -> list:
        # not accounting for putting the king in check or check-mate
        moves = []
        potential_moves = [[1, 0], [-1, 0], [0, 1], [1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for x, y in potential_moves:
            r, c = start_row + y, start_col + x
            # check that they are in the range of the board
            if r in range(8) and c in range(8):
                # if there is no piece in that spot or the piece in the spot is black, it can move there
                if not self.board[r][c] or self.board[r][c].color == Color.BLACK:
                    moves.append((r, c))
        return moves

    def get_piece(self, cell: str):
        col = cell[0]
        row = cell[1]
        alpha = 'abcdefgh'
        # now we have to get the right indices in our 2d array
        col = alpha.find(col)
        row = int(row) - 1
        return self.board[row][col]

    def show_board(self) -> None:
        d = dict()
        d[(PieceType.KING, Color.BLACK)] = "♚ "
        d[(PieceType.QUEEN, Color.BLACK)] = "♛ "
        d[(PieceType.ROOK, Color.BLACK)] = "♜ "
        d[(PieceType.BISHOP, Color.BLACK)] = "♝ "
        d[(PieceType.KNIGHT, Color.BLACK)] = "♞ "
        d[(PieceType.PAWN, Color.BLACK)] = "♟ "
        d[(PieceType.KING, Color.WHITE)] = "♔ "
        d[(PieceType.QUEEN, Color.WHITE)] = "♕ "
        d[(PieceType.ROOK, Color.WHITE)] = "♖ "
        d[(PieceType.BISHOP, Color.WHITE)] = "♗ "
        d[(PieceType.KNIGHT, Color.WHITE)] = "♘ "
        d[(PieceType.PAWN, Color.WHITE)] = "♙ "
        d[None] = "  "
        print("    a  b  c  d  e  f  g  h")
        print("____________________________")
        for rank in range(8):
            print(str(rank + 1) + "  |", end="")
            for file in range(8):
                print(d[self.board[rank][file]], end="|", )
            print("\n----------------------------|")
        print("\n\n\n")

    def possible_moves(self, source=None, index=None) -> bool:
        # this can either take a string such as b4 or a list index such as [2,1]
        if source:
            piece = self.get_piece(source)
        elif index:
            piece = self.board[index[0]][index[1]]

        if not piece:
            return False
        if source:
            start_col = source[0]
            start_row = source[1]
            alpha = 'abcdefgh'
            # now we have to get the right indices in our 2d array
            start_col = alpha.find(start_col)
            start_row = int(start_row) - 1

        elif index:
            start_row, start_col = index[0] , index[1]



        # eval will call the function that corresponds to the piece that needs to move
        if piece.piece_type == PieceType.PAWN:
            if piece.color == Color.WHITE:
                return self.move_WP(start_col, start_row)
            else:
                return self.move_BP(start_col, start_row)
        elif piece.piece_type == PieceType.KING:
            if piece.color == Color.WHITE:
                return self.move_WK(start_col, start_row)
            else:
                return self.move_BK(start_col, start_row)
        elif piece.piece_type == PieceType.BISHOP:
            if piece.color == Color.WHITE:
                return self.move_WP(start_col, start_row)
            else:
                return self.move_BP(start_col, start_row)
        elif piece.piece_type == PieceType.KNIGHT:
            if piece.color == Color.WHITE:
                return self.move_WN(start_col, start_row)
            else:
                return self.move_BN(start_col, start_row)
        elif piece.piece_type == PieceType.KNIGHT:
            if piece.color == Color.WHITE:
                return self.move_WR(start_col, start_row)
            else:
                return self.move_BR(start_col, start_row)
        elif piece.piece_type == PieceType.QUEEN:
            if piece.color == Color.WHITE:
                return self.move_WQ(start_col, start_row)
            else:
                return self.move_BQ(start_col, start_row)

    def move(self, source=None, target=None, source_index=None, target_index=None) -> bool:

        # function returns a boolean indicating if the move was successful
        possible = self.possible_moves(source, source_index)

        if not source_index and not target_index:
            start_col, start_row = source[0], source[1]
            end_col, end_row = target[0], target[1]

            # now we have to get the right indices in our 2d array
            alpha = 'abcdefgh'
            end_col, end_row = alpha.find(end_col), int(end_row) - 1
            start_col, start_row = alpha.find(start_col), int(start_row) - 1

        else:
            start_col, start_row = source_index[0], source_index[1]
            end_col, end_row = target_index[0], target_index[1]

        # if the starting spot is empty --> we cannot move
        if not self.board[start_row][start_col]:
            return None

        # this is where the player would like to move
        ans = (end_row, end_col)

        if ans in possible:  # the move is possible so do the move
            # if you are getting the opponent out, update the opponent score
            if self.board[end_row][end_col]:
                self.set_score(self.board[start_row][start_col].color, self.board[end_row][end_col].value)
            self.board[end_row][end_col] = self.board[start_row][start_col]
            self.board[start_row][start_col] = None
            return True
        else:
            return False

    def set_piece(self, location: str, piece_type: ChessPiece):
        file = ord(location[0]) - 97
        rank = int(location[1]) - 1
        self.board[rank][file] = piece_type

    def set_score(self, color, change) -> bool:
        if color == Color.WHITE:
            self.white_score -= change
            return True
        elif color == Color.BLACK:
            self.black_score -= change
            return True
        return False

    def find_move(self, turn_color):
        highest = 0
        target = ()
        source = ()

        # let's build up a list of where all the pieces in that color are located
        sources = []
        for row in range(8):
            for col in range(8):
                spot = self.board[row][col]
                if spot and spot.color == turn_color:
                    sources.append([row, col])
        for spot in sources:
            possible = self.possible_moves(None, spot)
            for row, col in possible:
                spot = self.board[row][col]
                if spot:
                    if not target and not source:
                        source = (spot[0], spot[1])
                        target = (row, col)
                    if spot.piece_type.value > highest:
                        highest = spot.piece_type.value
                        target = (row, col)
                return source, target

    def move_best_spot(self, turn_color):
        source, target = self.find_move(turn_color)

        removed = self.board[target[0]][target[1]]

        if removed:
            self.set_score(turn_color, removed.value)

        self.board[target[0]][target[1]] = self.board[source[0]][source[1]]
        self.board[source[0]][source[1]] = None


b = ChessBoard()
b.show_board()
print(b.move('h2', 'h4'))
b.show_board()
# this is an illegal move
print(b.move('a2', 'a5'))
print(b.move('b1', 'c3'))
b.show_board()
# make a white pawn capture its own kind aka an illegal move
print(b.move('b2', 'c3'))
# make a white pawn jump over
print(b.move('c2', 'c4'))
