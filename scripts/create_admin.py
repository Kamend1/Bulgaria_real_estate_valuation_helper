"""
One-time script to create the first admin user.

Usage:
    python -m scripts.create_admin
"""
import getpass
import sys

sys.path.insert(0, ".")

from app.db.models import User
from app.db.session import db_session
from app.services.auth_service import create_user, get_user_by_email, get_user_by_username


def main() -> None:
    print("=== Създаване на администраторски акаунт ===\n")

    email = input("Имейл: ").strip()
    if not email:
        print("Грешка: имейлът не може да е празен.")
        sys.exit(1)

    username = input("Потребителско ime: ").strip()
    if not username:
        print("Грешка: потребителското ime не може да е празно.")
        sys.exit(1)

    full_name = input("Пълно ime (по избор): ").strip()

    password = getpass.getpass("Парола: ")
    if len(password) < 8:
        print("Грешка: паролата трябва да е поне 8 символа.")
        sys.exit(1)

    password2 = getpass.getpass("Потвърди паролата: ")
    if password != password2:
        print("Грешка: паролите не съвпадат.")
        sys.exit(1)

    with db_session() as db:
        if get_user_by_email(db, email):
            print(f"Грешка: имейл '{email}' вече е регистриран.")
            sys.exit(1)
        if get_user_by_username(db, username):
            print(f"Грешка: потребителско ime '{username}' вече е заето.")
            sys.exit(1)

        user = create_user(
            db,
            email=email,
            username=username,
            password=password,
            full_name=full_name,
            role="admin",
        )
        db.commit()
        print(f"\n✓ Администраторски акаунт създаден успешно.")
        print(f"  ID: {user.id}")
        print(f"  Имейл: {user.email}")
        print(f"  Потребителско ime: {user.username}")


if __name__ == "__main__":
    main()
